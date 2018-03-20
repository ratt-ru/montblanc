#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace montblanc {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

using namespace tensorflow;

class QueuedTensorDatasetOp : public DatasetOpKernel {
private:
    DataTypeVector dtypes_;
    std::vector<TensorShape> shapes_;

public:
  explicit QueuedTensorDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx)
    {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_types", &dtypes_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_shapes", &shapes_));
    }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override
  {
    std::vector<PartialTensorShape> partial_shapes;

    for(int s=0; s < shapes_.size(); ++s)
    {
      PartialTensorShape partial_shape;
      const auto & shape = shapes_[s];
      const auto & dtype = dtypes_[s];
      for(int r=0; r < shape.dims(); ++r)
      {
        partial_shape.AddDim(shape.dim_size(r));
      }
      partial_shapes.emplace_back(partial_shape);
    }

    *output = new Dataset(ctx, dtypes_, partial_shapes);
  }

 private:
  class Dataset : public GraphDatasetBase {
   private:
    std::vector<PartialTensorShape> shapes;
    DataTypeVector dtypes;

   public:
    Dataset(OpKernelContext* ctx,
            const DataTypeVector & dtypes_,
            const std::vector<PartialTensorShape> & shapes_)
        : GraphDatasetBase(ctx), dtypes(dtypes_), shapes(shapes_) {}

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::SimpleQueue")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes;
    }

    string DebugString() override {
      return strings::StrCat("QueuedTensorDatasetOp()::Dataset");
    }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {

      AttrValue output_types;
      b->BuildAttrValue(dtypes, &output_types);
      AttrValue output_shapes;
      b->BuildAttrValue(shapes, &output_shapes);

      TF_RETURN_IF_ERROR(b->AddDataset(this, {},
                   {{"Toutput_types", output_types},
                    {"output_shapes", output_shapes}},
                   output));


      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        const auto & shapes = dataset()->shapes;
        const auto & dtypes = dataset()->dtypes;

        std::vector<TensorShape> max_shapes;

        for(int i = 0; i < dtypes.size(); ++i)
        {
          const PartialTensorShape& shape = shapes[i];
          TensorShape out_shape;

          for (int d = 0; d < shape.dims(); ++d)
          {
            out_shape.AddDim(shape.dim_size(d));
          }

          max_shapes.push_back(std::move(out_shape));
        }

        for(int s=0; s < shapes.size(); ++s)
        {
          Tensor components(cpu_allocator(), dtypes[s], max_shapes[s]);
          // components.setConstant(s);
          out_tensors->emplace_back(std::move(components));
        }

        *end_of_sequence = false;

        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        // TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("next"), next_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("next"), &next_));
        return Status::OK();
      }

     private:
      mutex mu_;
      // int64 next_ GUARDED_BY(mu_);
    };
  };
};

REGISTER_OP("QueuedTensorDataset")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("Toutput_shapes: list(shape) >= 1")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);  // TODO(mrry): Validate that
                                                // `components` have shapes
                                                // compatible with
// `output_shapes`.
REGISTER_KERNEL_BUILDER(Name("QueuedTensorDataset").Device(DEVICE_CPU),
                        QueuedTensorDatasetOp);

}  // namespace

}  // namespace montblanc
