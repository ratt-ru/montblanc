#include <deque>

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

using namespace tensorflow;

class QueueResource : public ResourceBase
{
private:
    mutex mu_;

    condition_variable cv_ GUARDED_BY(mu_);
    std::deque<std::vector<Tensor>> entries_ GUARDED_BY(mu_);
    bool closed_ GUARDED_BY(mu_);

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;

public:
    explicit QueueResource(const DataTypeVector & dtypes,
                           const std::vector<PartialTensorShape> & shapes)
      : dtypes_(dtypes), shapes_(shapes), closed_(false)
    {
    }

    void close(void) LOCKS_EXCLUDED(mu_)
    {
        {
            mutex_lock l(mu_);
            closed_ = true;
        }

        // Notify all waiting consumers
        cv_.notify_all();
    }

    Status insert(std::vector<Tensor> tensors) LOCKS_EXCLUDED(mu_)
    {
        {
            mutex_lock l(mu_);

            if(closed_)
                { return errors::OutOfRange("Queue is closed"); }

            entries_.push_back(std::move(tensors));
        }

        // Notify a waiting consumer
        cv_.notify_one();

        return Status::OK();
    }

    Status pop(std::vector<Tensor> * out) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Wait if empty and not closed
        while(entries_.empty() && !closed_)
            { cv_.wait(l); }

        // Bail if empty and closed
        if(entries_.empty() && closed_)
            { return errors::OutOfRange("Queue is closed"); }

        // Pop the first entry and return it
        *out = std::move(entries_.front());
        entries_.pop_front();

        return Status::OK();
    }

    const DataTypeVector &
    output_dtypes() const
      { return dtypes_; }

    const std::vector<PartialTensorShape> &
    output_shapes() const
      { return shapes_; }

    string DebugString() override
      { return "QueueResource"; }

};

class DatasetQueueHandleOp : public OpKernel
{
private:
    mutex mu_;

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;

    ContainerInfo cinfo GUARDED_BY(mu_);
    bool initialised GUARDED_BY(mu_);

public:
    explicit DatasetQueueHandleOp(OpKernelConstruction * ctx)
                : OpKernel(ctx),
                  initialised(false)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_types", &dtypes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_shapes", &shapes_));
    }

    ~DatasetQueueHandleOp() override
    {
        if(cinfo.resource_is_private_to_kernel())
        {
            if(!cinfo.resource_manager()->Delete<QueueResource>(
                cinfo.container(), cinfo.name()).ok())
            {
              // Do nothing; the resource will have been deleted by session resets.
            }
        }
    }

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // If not initialised, get the resource manager
        // and create the QueueResource
        if(!initialised)
        {
            ResourceMgr * mgr = ctx->resource_manager();
            OP_REQUIRES_OK(ctx, cinfo.Init(mgr, def()));

            QueueResource * resource;
            OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<QueueResource>(
                cinfo.container(), cinfo.name(), &resource,
                [this, ctx](QueueResource ** result) EXCLUSIVE_LOCKS_REQUIRED(mu_)
                {
                    *result = new QueueResource(dtypes_, shapes_);
                    return Status::OK();
                }
            ));

            initialised = true;
        }

        // Now assign the QueueResource to output position 0
        OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                  ctx, 0, cinfo.container(), cinfo.name(),
                  MakeTypeIndex<QueueResource>()));
    }
};

REGISTER_OP("DatasetQueueHandle")
    .Output("queue_handle: resource")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("Toutput_shapes: list(shape) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("DatasetQueueHandle")
                        .Device(DEVICE_CPU),
                        DatasetQueueHandleOp);

class DatasetQueueEnqueueOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit DatasetQueueEnqueueOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        QueueResource * queue_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &queue_resource));

        core::ScopedUnref unref_queue(queue_resource);

        // Convert component Tensors into a vector
        OpInputList components;
        OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));

        std::vector<Tensor> tensors;
        for (int c = 0; c < components.size(); ++c)
            { tensors.emplace_back(std::move(components[c])); }

        // Insert
        OP_REQUIRES_OK(ctx, queue_resource->insert(std::move(tensors)));
    }
};

REGISTER_OP("DatasetQueueEnqueue")
    .Input("queue_handle: resource")
    .Input("components: Toutput_types")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("DatasetQueueEnqueue")
                        .Device(DEVICE_CPU),
                        DatasetQueueEnqueueOp);


class QueueCloseOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit QueueCloseOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Obtain queue resource and close it
        QueueResource * queue_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &queue_resource));

        core::ScopedUnref unref_queue(queue_resource);

        queue_resource->close();
    }
};

REGISTER_OP("DatasetQueueClose")
    .Input("queue_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("DatasetQueueClose")
                        .Device(DEVICE_CPU),
                        QueueCloseOp);


// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class QueueDatasetOp : public DatasetOpKernel
{
public:
    explicit QueueDatasetOp(OpKernelConstruction * ctx)
                    : DatasetOpKernel(ctx) {}

    void MakeDataset(OpKernelContext * ctx, DatasetBase ** output) override
    {
        QueueResource * queue_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &queue_resource));

        core::ScopedUnref unref_queue(queue_resource);

        *output = new Dataset(ctx, queue_resource);
        // TODO(sjperkins)
        // Sometimes this is needed if kind of nothing is associated
        // with the dataset (iterators and next operators???????
        //(*output)->Ref();
    }

private:
  class Dataset : public GraphDatasetBase
  {
  public:
      QueueResource * queue_resource_;

      explicit Dataset(OpKernelContext * ctx, QueueResource * queue_resource)
          : GraphDatasetBase(ctx),
            queue_resource_(queue_resource)
      {
         queue_resource_->Ref();
      }

    ~Dataset() override
        { queue_resource_->Unref(); }

    const DataTypeVector & output_dtypes() const override
        { return queue_resource_->output_dtypes(); }

    const std::vector<PartialTensorShape> & output_shapes() const override
        { return queue_resource_->output_shapes(); }

    string DebugString()
        { return "QueueDataset"; }

    std::unique_ptr<IteratorBase>
    MakeIterator(const string & prefix) const override
    {
        return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::QueueDataset")}));
    }
  };


  class Iterator : public DatasetIterator<Dataset>
  {
  private:

  public:
    explicit Iterator(const Params & params)
        : DatasetIterator<Dataset>(params) {}

        virtual Status GetNextInternal(IteratorContext * ctx,
                    std::vector<Tensor> * out_tensors,
                    bool * end_of_sequence) override
        {
            *end_of_sequence = !dataset()->queue_resource_
                                         ->pop(out_tensors).ok();
            return Status::OK();
        }
  };
};

REGISTER_OP("QueueDataset")
    .Input("queue_handle: resource")
    .Output("handle: variant")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("QueueDataset").Device(DEVICE_CPU),
                        QueueDatasetOp);

}  // namespace

}  // namespace montblanc
