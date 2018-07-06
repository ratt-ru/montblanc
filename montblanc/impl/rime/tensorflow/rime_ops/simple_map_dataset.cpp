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

class MapResource : public ResourceBase
{
private:
    using Tuple = std::vector<Tensor>;
    using KeyType = int64;
    using MapType = std::unordered_map<KeyType, Tuple>;

private:
    mutex mu_;

    condition_variable cv_ GUARDED_BY(mu_);
    bool closed_ GUARDED_BY(mu_);
    MapType maps_ GUARDED_BY(mu_);

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;
    bool store_;

public:
    explicit MapResource(const DataTypeVector & dtypes,
                           const std::vector<PartialTensorShape> & shapes,
                           bool store)
      : dtypes_(dtypes), shapes_(shapes),
        store_(store), closed_(false)
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

        // Notify all waiting storers
        cv_.notify_all();
    }

    Status insert(const Tensor & tensor_key,
                  const Tuple & tensors) LOCKS_EXCLUDED(mu_)
    {
        int64 key = tensor_key.scalar<int64>()();

        // Slightly more optimal to release the lock
        // before the notify
        {
            mutex_lock l(mu_);

            if(closed_)
                { return errors::OutOfRange("Map is closed"); }

            maps_.insert({key, tensors});
        }

        // Notify a waiting storer
        cv_.notify_all();

        return Status::OK();
    }

    Status pop(const Tensor & tensor_key,
               std::vector<Tensor> * out) LOCKS_EXCLUDED(mu_)
    {
        int64 key = tensor_key.scalar<int64>()();

        mutex_lock l(mu_);

        while(true)
        {
            auto map_it = maps_.find(key);

            if(map_it != maps_.end())
            {
                // get
                if(store_)
                {
                    *out = map_it->second;
                }
                // consume
                else
                {
                    *out = std::move(map_it->second);
                    maps_.erase(map_it);
                }

                return Status::OK();
            }
            else if(closed_)
            {
                return errors::OutOfRange("Map is closed and empty");
            }

            // Wait for better conditions
            cv_.wait(l);
        }

        return errors::Internal("Should never exit pop while loop");
    }


    Status size(std::vector<int> * sizes) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        sizes->clear();
        sizes->push_back(maps_.size());

        return Status::OK();
    }

    Status keys(std::vector<int64> * keys) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        keys->clear();

        for(auto & value : maps_)
            { keys->push_back(value.first); }

        return Status::OK();
    }


    Status clear(const Tensor & tensor_keys) LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        if(tensor_keys.dims() == 0)
        {
            maps_.clear();
            return Status::OK();
        }

        auto keys = tensor_keys.tensor<int64, 1>();

        for(int i=0; i < tensor_keys.dim_size(0); ++i)
            { maps_.erase(keys(i)); }

        return Status::OK();
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
    bool store;

    ContainerInfo cinfo GUARDED_BY(mu_);
    bool initialised GUARDED_BY(mu_);

public:
    explicit DatasetMapHandleOp(OpKernelConstruction * ctx)
                : OpKernel(ctx),
                  initialised(false)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_types", &dtypes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_shapes", &shapes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("store", &store));
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
                    *result = new MapResource(dtypes_, shapes_, store);
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
    .Output("maps_handle: resource")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("Toutput_shapes: list(shape) >= 1")
    .Attr("store: bool = false")
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
    .Input("maps_handle: resource")
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
    .Input("maps_handle: resource")
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
        std::vector<int> sizes;
        OP_REQUIRES_OK(ctx, map_resource->size(&sizes));

        // Allocate size output tensor
        Tensor* size_ptr = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                            TensorShape({int(sizes.size())}), &size_ptr));

        auto size = size_ptr->tensor<int, 1>();

        for(int i=0; i < sizes.size(); ++i)
            { size(i) = sizes[i]; }
    }
};

REGISTER_OP("DatasetMapSize")
    .Input("maps_handle: resource")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("DatasetMapSize")
                        .Device(DEVICE_CPU),
                        MapSizeOp);



class MapKeysOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit MapKeysOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Obtain map resource and close it
        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &map_resource));

        core::ScopedUnref unref_map(map_resource);

        // Allocate size output tensor
        std::vector<int64> keys;
        OP_REQUIRES_OK(ctx, map_resource->keys(&keys));

        // Allocate size output tensor
        Tensor* key_ptr = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                            TensorShape({int(keys.size())}), &key_ptr));

        auto key = key_ptr->tensor<int, 1>();

        for(int i=0; i < keys.size(); ++i)
            { key(i) = keys[i]; }
    }
};


REGISTER_OP("DatasetMapKeys")
    .Input("maps_handle: resource")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("DatasetMapKeys")
                        .Device(DEVICE_CPU),
                        MapKeysOp);



class MapClearOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit MapClearOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);

        // Obtain map resource and close it
        MapResource * map_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &map_resource));

        core::ScopedUnref unref_map(map_resource);

        OP_REQUIRES_OK(ctx, map_resource->clear(ctx->input(1)));
    }
};

REGISTER_OP("DatasetMapClear")
    .Input("maps_handle: resource")
    .Input("keys: int64")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("DatasetMapClear")
                        .Device(DEVICE_CPU),
                        MapClearOp);



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
            std::size_t id;

        public:
            explicit Iterator(const Params & params)
                : DatasetIterator<Dataset>(params),
                  id(std::hash<Iterator *>{}(this))
            {
                // printf("Creating MapDataset::Iterator %p\n", (void *) this);
            }

            ~Iterator() override
            {
                // printf("Destroying MapDataset::Iterator %p\n", (void *) this);
            }

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
                Status status;
                std::vector<Tensor> keys;
                auto map_resource = dataset()->map_resource_;

                TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &keys,
                                                    end_of_sequence));

                // Nothing left in the input iterator
                if(*end_of_sequence)
                    { return Status::OK(); }

                // Insist on a single key
                if(keys.size() != 1)
                {
                    return errors::InvalidArgument("Got multiple keys (",
                                                    keys.size(),
                                                    "), expected 1.");
                }

                // Retrieve tensors from the map
                status = map_resource->pop(keys[0], out_tensors);

                if(!status.ok())
                {
                    if(!errors::IsOutOfRange(status))
                        { return status; }

                    // OutOfRange, indicate eos
                    *end_of_sequence = true;
                }

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
    .Input("maps_handle: resource")
    .Output("handle: variant")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("SimpleMapDataset").Device(DEVICE_CPU),
                        SimpleMapDatasetOp);

}  // namespace

}  // namespace montblanc
