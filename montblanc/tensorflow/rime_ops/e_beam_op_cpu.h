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
    CT & pol_sum,
    FT & abs_sum,
    typename tensorflow::TTypes<CT, 4>::ConstTensor & e_beam,
    const FT & gl, const FT & gm, const FT & gchan,
    int beam_lw, int beam_mh, int beam_nud, int pol,
    const FT & weight)
{
    if(gl < 0 || gl > beam_lw || gm < 0 || gm > beam_mh)
        { return; }

    CT data = e_beam(int(gl), int(gm), int(gchan), pol);
    abs_sum += weight*std::abs(data);
    pol_sum += data*FT(weight);
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
        const tf::Tensor & in_ebeam = context->input(6);

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

        OP_REQUIRES(context, in_ebeam.dims() == 4
            && in_ebeam.dim_size(3) == 4,
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

        cdata.nchan = in_point_errors.dim_size(2);
        cdata.npol = EBEAM_NPOL;
        cdata.npolchan = cdata.npol * cdata.nchan;

        cdata.beam_lw = in_ebeam.dim_size(0);
        cdata.beam_mh = in_ebeam.dim_size(1);
        cdata.beam_nud = in_ebeam.dim_size(2);

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
            cdata.nchan, EBEAM_NPOL});

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
        auto e_beam = in_ebeam.tensor<CT, 4>();
        auto jones = jones_ptr->tensor<CT, 5>();

        FT lmax = FT(cdata.beam_lw-1);
        FT mmax = FT(cdata.beam_mh-1);
        FT fmax = FT(cdata.beam_nud-1);
        constexpr FT zero = 0;

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

                    for(int chan=0; chan < cdata.nchan; chan++)
                    {
                        // Offset lm coordinates by point errors
                        // and scale by antenna scaling
                        FT vl = l + point_errors(time, ant, chan, 0);
                        FT vm = m + point_errors(time, ant, chan, 1);

                        vl *= antenna_scaling(ant, chan, 0);
                        vm *= antenna_scaling(ant, chan, 1);

                        // Shift into the cube coordinate system
                        vl = lscale*(vl - cdata.ll);
                        vm = mscale*(vm - cdata.lm);
                        FT vchan = fscale*(frequency(chan) - cdata.lf);

                        vl = std::max(zero, std::min(vl, lmax));
                        vm = std::max(zero, std::min(vm, mmax));
                        vchan = std::max(zero, std::min(vchan, fmax));

                        // Find the snapped grid coordinates
                        FT gl0 = std::floor(vl);
                        FT gm0 = std::floor(vm);
                        FT gchan0 = std::floor(vchan);

                        FT gl1 = std::min(FT(gl0+1.0), lmax);
                        FT gm1 = std::min(FT(gm0+1.0), mmax);
                        FT gchan1 = std::min(FT(gchan0+1.0), fmax);

                        // Difference between grid and offset coordinates
                        FT ld = vl - gl0;
                        FT md = vm - gm0;
                        FT chd = vchan - gchan0;

                        for(int pol=0; pol<EBEAM_NPOL; ++pol)
                        {
                            std::complex<FT> pol_sum = {0.0, 0.0};
                            FT abs_sum = 0.0;

                            // Load in the complex values from the E beam
                            // at the supplied coordinate offsets.
                            // Save the complex sum in pol_sum
                            // and the sum of abs in abs_sum
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm0, gchan0,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm0, gchan0,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm1, gchan0,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm1, gchan0,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*(1.0f-chd));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm0, gchan1,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm0, gchan1,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*(1.0f-md)*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm1, gchan1,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                (1.0f-ld)*md*chd);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm1, gchan1,
                                cdata.beam_lw, cdata.beam_mh, cdata.beam_nud, pol,
                                ld*md*chd);

                            // Normalising factor for the polarised sum
                            FT norm = 1.0 / std::abs(pol_sum);
                            if(!std::isfinite(norm))
                                { norm = 1.0; }

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