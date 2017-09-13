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
        const tf::Tensor & in_parallactic_angle_sin = context->input(4);
        const tf::Tensor & in_parallactic_angle_cos = context->input(5);
        const tf::Tensor & in_beam_extents = context->input(6);
        const tf::Tensor & in_beam_freq_map = context->input(7);
        const tf::Tensor & in_ebeam = context->input(8);

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int ntime = in_point_errors.dim_size(0);
        int na = in_point_errors.dim_size(1);

        int nchan = in_point_errors.dim_size(2);
        int npol = EBEAM_NPOL;
        int npolchan = npol * nchan;

        int beam_lw = in_ebeam.dim_size(0);
        int beam_mh = in_ebeam.dim_size(1);
        int beam_nud = in_ebeam.dim_size(2);

        // Extract beam extents
        auto beam_extents = in_beam_extents.tensor<FT, 1>();

        FT lower_l = beam_extents(0); // Lower l
        FT lower_m = beam_extents(1); // Lower m
        // FT lower_f = beam_extents(2); // Lower freq (unused)
        FT upper_l = beam_extents(3); // Upper l
        FT upper_m = beam_extents(4); // Upper m
        // FT upper_f = beam_extents(5); // Upper freq (unused)

        FT lscale = FT(beam_lw-1)/(upper_l - lower_l);
        FT mscale = FT(beam_mh-1)/(upper_m - lower_m);

        // Reason about our output shape
        tf::TensorShape jones_shape({nsrc,
            ntime, na, nchan, EBEAM_NPOL});

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
        auto parallactic_angle_sin = in_parallactic_angle_sin.tensor<FT, 2>();
        auto parallactic_angle_cos = in_parallactic_angle_cos.tensor<FT, 2>();
        auto beam_freq_map = in_beam_freq_map.flat<FT>();
        auto beam_freq_map_begin = beam_freq_map.data();
        auto beam_freq_map_end = beam_freq_map_begin + beam_freq_map.size();
        auto e_beam = in_ebeam.tensor<CT, 4>();
        auto jones = jones_ptr->tensor<CT, 5>();

        constexpr FT zero = 0.0;
        constexpr FT one = 1.0;

        FT lower_f = beam_freq_map(0);
        FT upper_f = beam_freq_map(beam_freq_map.size()-1);

        FT lmax = FT(beam_lw - one);
        FT mmax = FT(beam_mh - one);
        std::size_t fmax = beam_nud - 1;

        // Precompute channel dimension data
        std::vector<FT> gchan0(nchan);
        std::vector<FT> gchan1(nchan);
        std::vector<FT> chd0(nchan);
        std::vector<FT> chd1(nchan);

        #pragma omp parallel for
        for(int chan=0; chan < nchan; chan++)
        {
            // Get frequency and clamp to extents of the beam cube
            FT f = frequency(chan);
            f = std::min(f, upper_f);
            f = std::max(lower_f, f);

            // This really should work, but for beam_freq_map[i] < f < beam_freq_map[i+1]
            // it returns i+1...
            // TODO: understand why
            // std::size_t lchan = std::lower_bound(
            //     beam_freq_map_begin,
            //     beam_freq_map_end, f) - beam_freq_map_begin;

            std::size_t uchan = std::upper_bound(
                beam_freq_map_begin,
                beam_freq_map_end, f) - beam_freq_map_begin;

            uchan = std::min(uchan, fmax);

            std::size_t lchan = std::max(std::size_t(0), uchan - 1);

            FT lower_freq = *(beam_freq_map_begin + lchan);
            FT upper_freq = *(beam_freq_map_begin + uchan);
            FT freq_diff = upper_freq - lower_freq;

            gchan0[chan] = FT(lchan);
            gchan1[chan] = FT(uchan);
            chd0[chan] = (upper_freq - f)/freq_diff;
            chd1[chan] = (f - lower_freq)/freq_diff;

            // FT value = chd0[chan]*lower_freq + chd1[chan]*upper_freq;

            // printf("lfreq %.2f freq %.2f ufreq %.2f "
            //     "lgrid %.1f ugrid %.1f "
            //     "ldiff %.2f udiff %.2f "
            //     "value %.2f\n diff %.2f\n",
            //         lower_freq, f, upper_freq,
            //         gchan0[chan], gchan1[chan],
            //         chd0[chan], chd1[chan],
            //         value, value-f);
        }

        #pragma omp parallel for collapse(2)
        for(int time=0; time < ntime; ++time)
        {
            for(int ant=0; ant < na; ++ant)
            {
                // Rotation angle
                const FT & sint = parallactic_angle_sin(time, ant);
                const FT & cost = parallactic_angle_cos(time, ant);

                for(int src=0; src < nsrc; ++src)
                {
                    // Rotate lm coordinate angle
                    FT l = lm(src,0)*cost - lm(src,1)*sint;
                    FT m = lm(src,0)*sint + lm(src,1)*cost;

                    for(int chan=0; chan < nchan; chan++)
                    {
                        // Offset lm coordinates by point errors
                        // and scale by antenna scaling
                        FT vl = l + point_errors(time, ant, chan, 0);
                        FT vm = m + point_errors(time, ant, chan, 1);

                        vl *= antenna_scaling(ant, chan, 0);
                        vm *= antenna_scaling(ant, chan, 1);

                        // Shift into the cube coordinate system
                        vl = lscale*(vl - lower_l);
                        vm = mscale*(vm - lower_m);

                        vl = std::max(zero, std::min(vl, lmax));
                        vm = std::max(zero, std::min(vm, mmax));

                        // Find the snapped grid coordinates
                        FT gl0 = std::floor(vl);
                        FT gm0 = std::floor(vm);

                        FT gl1 = std::min(FT(gl0+one), lmax);
                        FT gm1 = std::min(FT(gm0+one), mmax);

                        // Difference between grid and offset coordinates
                        FT ld = vl - gl0;
                        FT md = vm - gm0;

                        for(int pol=0; pol<EBEAM_NPOL; ++pol)
                        {
                            std::complex<FT> pol_sum = {zero, zero};
                            FT abs_sum = zero;

                            // Load in the complex values from the E beam
                            // at the supplied coordinate offsets.
                            // Save the complex sum in pol_sum
                            // and the sum of abs in abs_sum
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm0, gchan0[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                (one-ld)*(one-md)*(chd0[chan]));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm0, gchan0[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                ld*(one-md)*(chd0[chan]));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm1, gchan0[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                (one-ld)*md*(chd0[chan]));
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm1, gchan0[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                ld*md*(chd0[chan]));

                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm0, gchan1[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                (one-ld)*(one-md)*chd1[chan]);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm0, gchan1[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                ld*(one-md)*chd1[chan]);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl0, gm1, gchan1[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                (one-ld)*md*chd1[chan]);
                            trilinear_interpolate<FT, CT>(pol_sum, abs_sum, e_beam,
                                gl1, gm1, gchan1[chan],
                                beam_lw, beam_mh, beam_nud, pol,
                                ld*md*chd1[chan]);

                            // Normalising factor for the polarised sum
                            FT norm = one / std::abs(pol_sum);
                            if(!std::isfinite(norm))
                                { norm = one; }

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
