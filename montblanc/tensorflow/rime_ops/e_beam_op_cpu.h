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
class EBeam<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit EBeam(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_frequency = context->input(1);
        const tf::Tensor & in_point_errors = context->input(2);
        const tf::Tensor & in_antenna_scaling = context->input(3);
        const tf::Tensor & in_parallactic_angle = context->input(4);
        const tf::Tensor & in_beam_extents = context->input(5);
        const tf::Tensor & in_E_beam = context->input(6);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument("lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_frequency.dims() == 1,
            tf::errors::InvalidArgument("frequency should be of shape (nchan,)"))

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

        OP_REQUIRES(context, in_parallactic_angle.dims() == 2,
            tf::errors::InvalidArgument("parallactic_angle should be of shape "
                                        "(ntime, na)"))

        OP_REQUIRES(context, in_beam_extents.dims() == 1
            && in_beam_extents.dim_size(0) == 6,
            tf::errors::InvalidArgument("beam_extents should be of shape "
                                        "(6,)"))

        // Constant data structure
        montblanc::ebeam::const_data<FT> cdata;

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

        // Extract beam extents
        auto beam_extents = in_beam_extents.tensor<FT, 1>();

        cdata.ll = beam_extents(0); // Lower l
        cdata.lm = beam_extents(1); // Lower m
        cdata.lf = beam_extents(2); // Lower frequency
        cdata.ul = beam_extents(3); // Upper l
        cdata.um = beam_extents(4); // Upper m
        cdata.uf = beam_extents(5); // Upper frequency

        FT lscale = FT(cdata.beam_lw-1)/(cdata.ul - cdata.ll);
        FT mscale = FT(cdata.beam_mh-1)/(cdata.um - cdata.lm);
        FT fscale = FT(cdata.beam_nud-1)/(cdata.uf - cdata.lf);

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
        auto frequency = in_frequency.tensor<FT, 1>();
        auto point_errors = in_point_errors.tensor<FT, 4>();
        auto antenna_scaling = in_antenna_scaling.tensor<FT, 3>();
        auto parallactic_angle = in_parallactic_angle.tensor<FT, 2>();
        auto e_beam = in_E_beam.tensor<CT, 4>();
        auto jones = jones_ptr->tensor<CT, 5>();

        int chan_ext = cdata.nchan.extent_size();

        for(int time=0; time < cdata.ntime; ++time)
        {                
            for(int ant=0; ant < cdata.na; ++ant)
            {
                // Rotation angle
                FT angle = parallactic_angle(time, ant);
                FT sint = std::sin(angle);
                FT cost = std::cos(angle);

                for(int src=0; src < cdata.nsrc; ++src)
                {
                    // Rotate lm coordinate angle
                    FT l = lm(src,0)*cost - lm(src,1)*sint;
                    FT m = lm(src,0)*sint + lm(src,1)*cost;

                    for(int chan=0; chan < chan_ext; chan++)
                    {
                        // Offset lm coordinates by point errors
                        // and scale by antenna scaling
                        FT ol = l + point_errors(time, ant, chan, 0);
                        FT om = m + point_errors(time, ant, chan, 1);

                        ol *= antenna_scaling(ant, chan, 0);
                        om *= antenna_scaling(ant, chan, 1);

                        // Change into the cube coordinate system
                        ol = lscale*(ol - cdata.ll);
                        om = mscale*(om - cdata.lm);
                        FT ochan = fscale*(frequency(chan) - cdata.lf);

                        ol = std::max(FT(0.0), std::min(ol, FT(cdata.beam_lw-1)));
                        om = std::max(FT(0.0), std::min(om, FT(cdata.beam_mh-1)));
                        ochan = std::max(FT(0.0), std::min(ochan, FT(cdata.beam_nud-1)));

                        // Find the quantized grid coordinate of the offset coordinate
                        FT gl = std::floor(ol);
                        FT gm = std::floor(om);
                        FT gchan = std::floor(ochan);

                        // Difference between grid and offset coordinates
                        FT ld = ol - gl;
                        FT md = om - gm;
                        FT chd = ochan - gchan;

                        for(int pol=0; pol<EBEAM_NPOL; ++pol)
                        {
                            std::complex<FT> pol_sum = {0.0, 0.0};
                            FT abs_sum = 0.0;

                            // Load in the complex values from the E beam
                            // at the supplied coordinate offsets.
                            // Save the complex sum in pol_sum
                            // and the sum of abs in abs_sum
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 0.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 0.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 1.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 1.0f, gchan + 0.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 0.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 0.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 0.0f, gm + 1.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl + 1.0f, gm + 1.0f, gchan + 1.0f,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*chd);

                            // Normalising factor for the polarised sum
                            FT norm = 1.0/std::abs(pol_sum);
                            // Multiply in the absolute value
                            pol_sum.real(pol_sum.real() * norm * abs_sum);
                            pol_sum.imag(pol_sum.imag() * norm * abs_sum);
                            jones(src,time,ant,chan,pol) = pol_sum;
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