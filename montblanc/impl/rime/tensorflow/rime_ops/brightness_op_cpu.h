#ifndef RIME_BRIGHTNESS_OP_CPU_H
#define RIME_BRIGHTNESS_OP_CPU_H

#include "brightness_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the Brightness op for CPUs
template <typename FT, typename CT>
class Brightness<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit Brightness(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_stokes = context->input(0);

        // Allocate output tensors
        // Allocate space for output tensor 'brightness'
        tf::Tensor * brightness_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_stokes.shape(), &brightness_ptr));

        // Extract Eigen tensors
        auto stokes = in_stokes.flat_inner_dims<FT>();
        auto brightness = brightness_ptr->flat_inner_dims<CT>();

        auto nrows = stokes.dimension(0);
        auto npols = stokes.dimension(1);

        OP_REQUIRES(context, npols == 4,
            tf::errors::InvalidArgument("Polarisations must be '4'."));

        for(int r=0; r < nrows; ++r)
        {
            const auto & I = stokes(r, 0);
            const auto & Q = stokes(r, 1);
            const auto & U = stokes(r, 2);
            const auto & V = stokes(r, 3);

            brightness(r, 0) = {I + Q, 0.0};
            brightness(r, 1) = {U, V};
            brightness(r, 2) = {U, -V};
            brightness(r, 3) = {I - Q, 0.0};
        }

    }
};

MONTBLANC_BRIGHTNESS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_BRIGHTNESS_OP_CPU_H
