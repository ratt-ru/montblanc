#include <unordered_map>

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

// Partial Ordering Comparator for Tensor keys containing scalar int64's
struct KeyTensorLess {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::less<int64>{}(lhs.scalar<int64>()(), rhs.scalar<int64>()());
  }
};

// Key Equality operator for Tensor keys containing scalar int64's
struct KeyTensorEqual {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::equal_to<int64>{}(lhs.scalar<int64>()(), rhs.scalar<int64>()());
  }
};

// Hash for Tensor keys containing scalar int64's
struct KeyTensorHash {
  std::size_t operator()(const Tensor& key) const {
    return std::hash<int64>{}(key.scalar<int64>()());
  }
};

class MapResource : public ResourceBase
{
private:
    using Tuple = std::vector<Tensor>;
    using KeyType = Tensor;
    using MapType = std::unordered_map<KeyType, Tuple,
                                        KeyTensorHash, KeyTensorEqual>;

private:
    mutex mu_;

    condition_variable cv_ GUARDED_BY(mu_);
    bool closed_ GUARDED_BY(mu_);
    MapType map_ GUARDED_BY(mu_);

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;

public:
    explicit MapResource(const DataTypeVector & dtypes,
                           const std::vector<PartialTensorShape> & shapes)
      : dtypes_(dtypes), shapes_(shapes), closed_(false)
    {
        // printf("Creating MapResource %p\n", (void *) this);
    }

    ~MapResource() override
    {
        // printf("Destroying MapResource %p\n", (void *) this);
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

    Status insert(const KeyType & key, std::vector<Tensor> tensors) LOCKS_EXCLUDED(mu_)
    {
        {
            mutex_lock l(mu_);

            if(closed_)
                { return errors::OutOfRange("Map is closed"); }

            map_.insert({key, tensors});
        }

        // Notify a waiting consumer
        cv_.notify_one();

        return Status::OK();
    }

    Status pop(const KeyType & key, std::vector<Tensor> * out) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        typename MapType::iterator it;

        // Wait until the element with the requested key is present
        while(((it = map_.find(key)) == map_.end()) && !closed_)
            { cv_.wait(l); }

        if(it == map_.end() && closed_)
            { return errors::OutOfRange("Map is closed"); }

        *out  = std::move(it->second);
        map_.erase(it);

        return Status::OK();
    }


    std::size_t size(void) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);
        return map_.size();
    }


    const DataTypeVector &
    output_dtypes() const
      { return dtypes_; }

    const std::vector<PartialTensorShape> &
    output_shapes() const
      { return shapes_; }

    string DebugString() override
      { return "MapResource"; }

};

class DatasetMapHandleOp : public OpKernel
{
private:
    mutex mu_;

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;

    ContainerInfo cinfo GUARDED_BY(mu_);
    bool initialised GUARDED_BY(mu_);

public:
    explicit DatasetMapHandleOp(OpKernelConstruction * ctx)
                : OpKernel(ctx),
                  initialised(false)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_types", &dtypes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_shapes", &shapes_));
    }

    ~DatasetMapHandleOp() override
    {
        if(cinfo.resource_is_private_to_kernel())
        {
            if(!cinfo.resource_manager()->Delete<MapResource>(
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
        // and create the MapResource
        if(!initialised)
        {
            ResourceMgr * mgr = ctx->resource_manager();
            OP_REQUIRES_OK(ctx, cinfo.Init(mgr, def()));

            MapResource * map_resource;
            OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<MapResource>(
                cinfo.container(), cinfo.name(), &map_resource,
                [this, ctx](MapResource ** result) EXCLUSIVE_LOCKS_REQUIRED(mu_)
                {
                    *result = new MapResource(dtypes_, shapes_);
                    return Status::OK();
                }
            ));

            core::ScopedUnref unref_map(map_resource);

            initialised = true;
        }

        // Now assign the MapResource to output position 0
        OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                  ctx, 0, cinfo.container(), cinfo.name(),
                  MakeTypeIndex<MapResource>()));
    }
};

REGISTER_OP("DatasetMapHandle")
    .Output("map_handle: resource")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("Toutput_shapes: list(shape) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("DatasetMapHandle")
                        .Device(DEVICE_CPU),
                        DatasetMapHandleOp);

class DatasetMapInsertOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit DatasetMapInsertOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &map_resource));

        core::ScopedUnref unref_map(map_resource);

        const Tensor * key_tensor;
        OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));

        // Convert component Tensors into a vector
        OpInputList components;
        OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));

        std::vector<Tensor> tensors;
        for (int c = 0; c < components.size(); ++c)
            { tensors.emplace_back(std::move(components[c])); }

        // Insert
        OP_REQUIRES_OK(ctx, map_resource->insert(*key_tensor, std::move(tensors)));
    }
};

REGISTER_OP("DatasetMapInsert")
    .Input("map_handle: resource")
    .Input("key: int64")
    .Input("components: Toutput_types")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("DatasetMapInsert")
                        .Device(DEVICE_CPU),
                        DatasetMapInsertOp);


class MapCloseOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit MapCloseOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Obtain map resource and close it
        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &map_resource));

        core::ScopedUnref unref_map(map_resource);

        map_resource->close();
    }
};

REGISTER_OP("DatasetMapClose")
    .Input("map_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("DatasetMapClose")
                        .Device(DEVICE_CPU),
                        MapCloseOp);


class MapSizeOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit MapSizeOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Obtain map resource and close it
        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &map_resource));

        core::ScopedUnref unref_map(map_resource);

        // Allocate size output tensor
        Tensor* size = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

        // Set it to the actual size
        size->scalar<int32>().setConstant(map_resource->size());
    }
};

REGISTER_OP("DatasetMapSize")
    .Input("map_handle: resource")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("DatasetMapSize")
                        .Device(DEVICE_CPU),
                        MapSizeOp);




// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class SimpleMapDatasetOp : public DatasetOpKernel
{
public:
    explicit SimpleMapDatasetOp(OpKernelConstruction * ctx)
                    : DatasetOpKernel(ctx) {}

protected:
    void MakeDataset(OpKernelContext * ctx, DatasetBase ** output) override
    {
        DatasetBase * input;
        OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0),
                                                            &input));

        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1),
                                                        &map_resource));

        core::ScopedUnref unref_map(map_resource);

        *output = new Dataset(ctx, input, map_resource);
        // TODO(sjperkins)
        // Sometimes this is needed if kind of nothing is associated
        // with the dataset (iterators and next operators???????
        //(*output)->Ref();
    }

private:
    class Dataset : public GraphDatasetBase
    {
    public:
        const DatasetBase * input_;
        MapResource * map_resource_;

        explicit Dataset(OpKernelContext * ctx,
                        const DatasetBase * input,
                        MapResource * map_resource)
                : GraphDatasetBase(ctx),
                    input_(input),
                    map_resource_(map_resource)
        {
            input_->Ref();
            map_resource_->Ref();
            // printf("Creating MapDatset %p\n", (void *) this);
        }

        ~Dataset() override
        {
            input_->Unref();
            map_resource_->Unref();
            // printf("Destroying MapDatset %p\n", (void *) this);
        }


        Dataset(const Dataset & rhs) = delete;
        Dataset & operator=(const Dataset & rhs) = delete;

        const DataTypeVector & output_dtypes() const override
            { return map_resource_->output_dtypes(); }

        const std::vector<PartialTensorShape> & output_shapes() const override
            { return map_resource_->output_shapes(); }

        string DebugString() const
            { return "SimpleMapDataset"; }

        std::unique_ptr<IteratorBase>
        MakeIteratorInternal(const string & prefix) const override
        {
            return std::unique_ptr<IteratorBase>(new Iterator(
              {this, strings::StrCat(prefix, "::SimpleMapDataset")}));
        }

    protected:
        Status AsGraphDefInternal(OpKernelContext * ctx,
                                DatasetGraphDefBuilder * b,
                                Node ** output) const override
        {
            return errors::InvalidArgument("Not Implemented");
        }

    private:
        class Iterator : public DatasetIterator<Dataset>
        {
        private:
            mutex mu_;

            std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

        public:
            explicit Iterator(const Params & params)
                : DatasetIterator<Dataset>(params) {}

            Status Initialize(IteratorContext * ctx) override
            {
                return dataset()->input_->MakeIterator(ctx,
                                            prefix(),
                                            &input_impl_);
            }

            virtual Status GetNextInternal(IteratorContext * ctx,
                        std::vector<Tensor> * out_tensors,
                        bool * end_of_sequence) override
            {
                std::vector<Tensor> keys;

                TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &keys,
                                                    end_of_sequence));

                if(*end_of_sequence)
                    { return Status::OK(); }

                if(keys.size() != 1)
                {
                    return errors::InvalidArgument("Got multiple keys (",
                                                    keys.size(),
                                                    "), expected 1.");
                }

                *end_of_sequence = !dataset()->map_resource_
                                      ->pop(keys[0], out_tensors).ok();
                return Status::OK();
            }
        protected:
          Status SaveInternal(IteratorStateWriter* writer) override
            { return errors::InvalidArgument("Not Implemented"); }

          Status RestoreInternal(IteratorContext * ctx,
                                IteratorStateReader * reader) override
            { return errors::InvalidArgument("Not Implemented"); }
        }; // class Iterator
    };     // class Dataset
};         // class SimpleMapDatasetOp

REGISTER_OP("SimpleMapDataset")
    .Input("key_dataset: variant")
    .Input("map_handle: resource")
    .Output("handle: variant")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("SimpleMapDataset").Device(DEVICE_CPU),
                        SimpleMapDatasetOp);

}  // namespace

}  // namespace montblanc
