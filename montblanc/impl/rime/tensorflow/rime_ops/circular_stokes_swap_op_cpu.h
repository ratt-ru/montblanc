#ifndef RIME_CIRCULAR_STOKES_SWAP_OP_CPU_H
#define RIME_CIRCULAR_STOKES_SWAP_OP_CPU_H

#include "circular_stokes_swap_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the CircularStokesSwap op for CPUs
template <typename FT>
class CircularStokesSwap<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit CircularStokesSwap(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_stokes_in = context->input(0);
        
        int nsrc = in_stokes_in.dim_size(0);
        int ntime = in_stokes_in.dim_size(1);
        int npol = in_stokes_in.dim_size(2);


        // Extract Eigen tensors
        auto stokes_in = in_stokes_in.tensor<FT, 3>();
        

        // Allocate output tensors
        // Allocate space for output tensor 'stokes_out'
        tf::Tensor * stokes_out_ptr = nullptr;
        tf::TensorShape stokes_out_shape = tf::TensorShape({ nsrc, ntime, npol });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, stokes_out_shape, &stokes_out_ptr));

        auto stokes_out = stokes_out_ptr->tensor<FT, 3>();

        #pragma omp parallel for collapse(2)
        for(int src=0; src<nsrc; ++src)
        {
            for(int time=0; time<ntime; ++time)
            {
                stokes_out(src, time, 0) = stokes_in(src, time, 0);
                stokes_out(src, time, 1) = stokes_in(src, time, 3);
                stokes_out(src, time, 2) = stokes_in(src, time, 1);
                stokes_out(src, time, 3) = stokes_in(src, time, 2);
            }
        }
    }
};

MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CIRCULAR_STOKES_SWAP_OP_CPU_H