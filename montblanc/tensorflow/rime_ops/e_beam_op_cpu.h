#ifndef RIME_E_BEAM_OP_CPU_H_
#define RIME_E_BEAM_OP_CPU_H_

#include "e_beam_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace ebeam {
// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;    

template <typename FT, typename CT>
inline void
trilinear_interpolate(
    CT & sum,
    FT & abs_sum,
    typename tensorflow::TTypes<CT, 4>::ConstTensor & e_beam,
    float gl, float gm, float gchan,
    int beam_lw, int beam_mh, int beam_nud, int pol,
    float weight)
{
    if(gl < 0 || gl > beam_lw || gm < 0 || gm > beam_mh)
        { return; }

    CT data = e_beam(int(gl), int(gm), int(gchan), pol);
    abs_sum += weight*std::abs(data);
    sum += data*FT(weight);
}

template <typename FT, typename CT>
class RimeEBeam<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit RimeEBeam(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_point_errors = context->input(1);
        const tf::Tensor & in_antenna_scaling = context->input(2);
        const tf::Tensor & in_E_beam = context->input(3);
        const tf::Tensor & in_parallactic_angle = context->input(4);
        const tf::Tensor & in_beam_ll = context->input(5);
        const tf::Tensor & in_beam_lm = context->input(6);
        const tf::Tensor & in_beam_ul = context->input(7);
        const tf::Tensor & in_beam_um = context->input(8);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument("lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_point_errors.dims() == 4
            && in_point_errors.dim_size(3) == 2,
            tf::errors::InvalidArgument("point_errors should be of shape "
                                        "(ntime, na, nchan, 2)"))

        OP_REQUIRES(context, in_antenna_scaling.dims() == 3
            && in_antenna_scaling.dim_size(2) == 2,
            tf::errors::InvalidArgument("antenna_scaling should be of shape "
                                        "(na, nchan, 2)"))

        OP_REQUIRES(context, in_E_beam.dims() == 4
            && in_E_beam.dim_size(3) == 4,
            tf::errors::InvalidArgument("E_Beam should be of shape "
                                        "(beam_lw, beam_mh, beam_nud, 4)"))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_parallactic_angle.shape()),
            tf::errors::InvalidArgument("parallactic_angle is not scalar: ",
                in_parallactic_angle.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ll.shape()),
            tf::errors::InvalidArgument("in_beam_ll is not scalar: ",
                in_beam_ll.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_lm.shape()),
            tf::errors::InvalidArgument("in_beam_lm is not scalar: ",
                in_beam_lm.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ul.shape()),
            tf::errors::InvalidArgument("in_beam_ul is not scalar: ",
                in_beam_ul.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_um.shape()),
            tf::errors::InvalidArgument("in_beam_um is not scalar: ",
                in_beam_um.shape().DebugString()))

        // Constant data structure
        montblanc::ebeam::const_data cdata;

        // Extract problem dimensions
        cdata.nsrc = in_lm.dim_size(0);
        cdata.ntime = in_point_errors.dim_size(0);
        cdata.na = in_point_errors.dim_size(1);

        cdata.nchan.local_size = in_point_errors.dim_size(2);
        cdata.nchan.global_size = cdata.nchan.local_size;
        cdata.nchan.lower_extent = 0;
        cdata.nchan.upper_extent = cdata.nchan.local_size;

        cdata.npolchan.local_size = cdata.nchan.local_size*EBEAM_NPOL;
        cdata.npolchan.global_size = cdata.npolchan.local_size;
        cdata.npolchan.lower_extent = 0;
        cdata.npolchan.upper_extent = cdata.npolchan.local_size;

        cdata.beam_lw = in_E_beam.dim_size(0);
        cdata.beam_mh = in_E_beam.dim_size(1);
        cdata.beam_nud = in_E_beam.dim_size(2);

        // Reason about our output shape
        tf::TensorShape jones_shape({cdata.nsrc,
            cdata.ntime, cdata.na,
            cdata.nchan.local_size, EBEAM_NPOL});

        // Create a pointer for the jones result
        tf::Tensor * jones_ptr = nullptr;

        // Allocate memory for the jones
        OP_REQUIRES_OK(context, context->allocate_output(
            0, jones_shape, &jones_ptr));

        if (jones_ptr->NumElements() == 0)
            { return; }

        auto lm = in_lm.tensor<FT, 2>();
        auto point_errors = in_point_errors.tensor<FT, 4>();
        auto antenna_scaling = in_antenna_scaling.tensor<FT, 3>();
        auto e_beam = in_E_beam.tensor<CT, 4>();
        auto jones = jones_ptr->tensor<CT, 5>();

        FT parallactic_angle = in_parallactic_angle.tensor<FT, 0>()(0);
        FT beam_ll = in_beam_ll.tensor<FT, 0>()(0);
        FT beam_lm = in_beam_lm.tensor<FT, 0>()(0);
        FT beam_ul = in_beam_ul.tensor<FT, 0>()(0);
        FT beam_um = in_beam_um.tensor<FT, 0>()(0);

        int chan_ext = cdata.nchan.extent_size();

        for(int time=0; time < cdata.ntime; ++time)
        {
            // Rotation angle
            FT angle = parallactic_angle*time;
            FT sint = std::sin(angle);
            FT cost = std::cos(angle);

            for(int src=0; src < cdata.nsrc; ++src)
            {
                // Rotate lm coordinate angle
                FT l = lm(src,0)*cost - lm(src,1)*sint;
                FT m = lm(src,0)*sint + lm(src,1)*cost;
                
                for(int ant=0; ant < cdata.na; ++ant)
                {
                    for(int chan=0; chan < chan_ext; chan++)
                    {
                        // Offset lm coordinates by point errors
                        // and scale by antenna scaling
                        FT ol = l + point_errors(time, ant, chan, 0);
                        FT om = m + point_errors(time, ant, chan, 1);

                        ol *= antenna_scaling(ant, chan, 0);
                        om *= antenna_scaling(ant, chan, 1);

                        // Change into the cube coordinate system
                        ol = FT(cdata.beam_lw-1)*(ol - beam_ll)/(beam_ul - beam_ll);
                        om = FT(cdata.beam_mh-1)*(om - beam_lm)/(beam_um - beam_lm);
                        float ochan = float(cdata.beam_nud-1) *
                            float(chan+cdata.nchan.lower_extent) /
                            float(cdata.nchan.global_size);

                        // Find the quantized grid coordinate of the offset coordinate
                        float gl = std::floor(ol);
                        float gm = std::floor(om);
                        float gchan = std::floor(ochan);

                        // Difference between grid and offset coordinates
                        float ld = ol - gl;
                        float md = om - gm;
                        float chd = ochan - gchan;

                        for(int pol=0; pol<EBEAM_NPOL; ++pol)
                        {
                            std::complex<FT> sum = {0.0, 0.0};
                            FT abs_sum = 0.0;

                            // Load in the complex values from the E beam
                            // at the supplied coordinate offsets.
                            // Save the sum of abs in sum.real
                            // and the sum of args in sum.imag
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 0.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 0.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 1.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 1.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 0.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 0.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 1.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*chd);
                            trilinear_interpolate<FT, CT>(sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 1.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*chd);

                            // Find the angle for the given polarisation
                            FT sum_angle = std::arg(sum);
                            // The exponent of sum_angle*1j
                            FT real = std::cos(sum_angle)*abs_sum;
                            FT imag = std::sin(sum_angle)*abs_sum;
                            jones(src,time,ant,chan,pol) = {real, imag};
                        }
                    }
                }
            }
        }
    }
};

} // namespace ebeam {
} // namespace montblanc {

#endif // #ifndef RIME_E_BEAM_OP_CPU_H_