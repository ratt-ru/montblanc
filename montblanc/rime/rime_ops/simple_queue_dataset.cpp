#include <chrono>
#include <deque>
#include <unordered_map>

#include "absl/strings/str_join.h"

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
public:
    using Tuple = std::vector<Tensor>;
    using Queue = std::deque<Tuple>;

private:
    mutex mu_;

    condition_variable cv_ GUARDED_BY(mu_);
    Queue queue GUARDED_BY(mu_);
    bool closed_ GUARDED_BY(mu_);

    DataTypeVector dtypes_;
    std::vector<PartialTensorShape> shapes_;
    std::string name_;

public:
public:
    explicit QueueResource(const DataTypeVector & dtypes,
                           const std::vector<PartialTensorShape> & shapes,
                           const std::string & name)
      : dtypes_(dtypes), shapes_(shapes), name_(name), closed_(false)
    {
        printf("%s Creating QueueResource %p\n", name_.c_str(), (void *) this);
    }

    ~QueueResource() override
    {
        printf("%s Destroying QueueResource %p\n", name_.c_str(), (void *) this);
    }

    const DataTypeVector &
    output_dtypes() const
      { return dtypes_; }

    const std::vector<PartialTensorShape> &
    output_shapes() const
      { return shapes_; }

    std::string DebugString() override
      { return "QueueResource"; }


    void close(void) LOCKS_EXCLUDED(mu_)
    {
        {
            mutex_lock l(mu_);
            closed_ = true;
        }

        // Notify all waiting consumers
        cv_.notify_all();
    }

    Status insert(const Tuple & data) LOCKS_EXCLUDED(mu_)
    {
        // Slightly more optimal to unlock the mutex
        // before the notify
        printf("%s queue.inserting\n", name_.c_str());

        {
            mutex_lock l(mu_);


            if(closed_)
                { return errors::OutOfRange("Queue is closed"); }

            queue.push_back(data);

            printf("%s queue.inserted\n", name_.c_str());

        }


        // Notify waiting consumers
        cv_.notify_all();

        return Status::OK();
    }

    Status pop(Tuple * out) LOCKS_EXCLUDED(mu_)
    {
        while(true)
        {
            mutex_lock l(mu_);
            printf("%s queue.popwait\n", name_.c_str());

            if(!queue.empty())
            {
                // Pop the first entry and return it
                *out = std::move(queue.front());
                queue.pop_front();
                printf("%s queue.popped\n", name_.c_str());
                return Status::OK();
            }
            else if (closed_)
                { return errors::OutOfRange("Queue is closed and empty"); }

            // Wait for better conditions
            cv_.wait_for(l, std::chrono::seconds(2));
        }

        return errors::Internal("Should never exit pop while loop");
    }

    std::size_t size() LOCKS_EXCLUDED(mu_)
    {
        mutex_lock l(mu_);
        return queue.size();
    }
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
        // mutex_lock l(mu_);

        // If not initialised, get the resource manager
        // and create the QueueResource
        if(!initialised)
        {
            ResourceMgr * mgr = ctx->resource_manager();
            OP_REQUIRES_OK(ctx, cinfo.Init(mgr, def()));

            QueueResource * queue_resource;
            OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<QueueResource>(
                cinfo.container(), cinfo.name(), &queue_resource,
                [this, ctx](QueueResource ** result) EXCLUSIVE_LOCKS_REQUIRED(mu_)
                {
                    *result = new QueueResource(dtypes_, shapes_, cinfo.name());
                    return Status::OK();
                }
            ));

            core::ScopedUnref unref_queue(queue_resource);

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
        // mutex_lock l(mu_);

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
        // mutex_lock l(mu_);

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


class QueueSizeOp : public OpKernel
{
private:
    mutex mu_;

public:
    explicit QueueSizeOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext * ctx) override LOCKS_EXCLUDED(mu_)
    {
        // mutex_lock l(mu_);

        // Obtain queue resource and close it
        QueueResource * queue_resource;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                          &queue_resource));

        core::ScopedUnref unref_queue(queue_resource);

        // Allocate size output tensor
        Tensor* size_ptr = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                            TensorShape({}), &size_ptr));

        auto size = size_ptr->scalar<int>();
        size() = queue_resource->size();
    }
};

REGISTER_OP("DatasetQueueSize")
    .Input("queue_handle: resource")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("DatasetQueueSize")
                        .Device(DEVICE_CPU),
                        QueueSizeOp);




// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class SimpleQueueDatasetOp : public DatasetOpKernel
{
public:
    explicit SimpleQueueDatasetOp(OpKernelConstruction * ctx)
                    : DatasetOpKernel(ctx) {}

protected:
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
    class Dataset : public DatasetBase
    {
    public:
        QueueResource * queue_resource_;

        explicit Dataset(OpKernelContext * ctx, QueueResource * queue_resource)
                : DatasetBase(DatasetContext(ctx)),
                  queue_resource_(queue_resource)
        {
            printf("Creating QueueDataset %p\n", (void *) this);
            queue_resource_->Ref();
        }

        Dataset(const Dataset & rhs) = delete;
        Dataset & operator=(const Dataset & rhs) = delete;

        ~Dataset() override
        {
            printf("Destroying QueueDataset %p\n", (void *) this);
            queue_resource_->Unref();
        }

        const DataTypeVector & output_dtypes() const override
            { return queue_resource_->output_dtypes(); }

        const std::vector<PartialTensorShape> & output_shapes() const override
            { return queue_resource_->output_shapes(); }

        string DebugString() const override
            { return "SimpleQueueDataset"; }

        std::unique_ptr<IteratorBase>
        MakeIteratorInternal(const string & prefix) const override
        {
            return std::unique_ptr<IteratorBase>(new Iterator(
              {this, strings::StrCat(prefix, "::SimpleQueueDataset")}));
        }

    protected:
        Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** node) const override
        {
            return errors::Unimplemented("AsGraphDefInternal");
        }


    private:
        class Iterator : public DatasetIterator<Dataset>
        {
        public:
            explicit Iterator(const Params & params)
                : DatasetIterator<Dataset>(params)
            {
                printf("Creating QueueDataset::Iterator %p\n", (void *) this);
            }

            ~Iterator() override
            {
                printf("Destroying QueueDataset::Iterator %p\n", (void *) this);
            }

            Status GetNextInternal(IteratorContext * ctx,
                        std::vector<Tensor> * out_tensors,
                        bool * end_of_sequence) override
            {
                auto & queue = dataset()->queue_resource_;

                Status status = queue->pop(out_tensors);

                if(!status.ok())
                {
                    // OutOfRange, indicate eos
                    if(errors::IsOutOfRange(status))
                    {
                        *end_of_sequence = true;
                        return Status::OK();
                    }

                    return status;
                }

                return Status::OK();
            }
        protected:
          Status SaveInternal(IteratorStateWriter* writer) override
            {
                return errors::Unimplemented("SaveInternal");
            }

          Status RestoreInternal(IteratorContext * ctx,
                                IteratorStateReader * reader) override
            {
                return errors::Unimplemented("RestoreInternal");
            }
        }; // class Iterator
    };     // class Dataset
};         // class SimpleQueueDatasetOp

REGISTER_OP("SimpleQueueDataset")
    .Input("queue_handle: resource")
    .Output("handle: variant")
    .SetIsStateful()  // Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("SimpleQueueDataset").Device(DEVICE_CPU),
                        SimpleQueueDatasetOp);

}  // namespace

}  // namespace montblanc
