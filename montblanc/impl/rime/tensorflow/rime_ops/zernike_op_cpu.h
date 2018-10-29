#ifndef ZERNIKE_DDE_ZERNIKE_OP_CPU_H
#define ZERNIKE_DDE_ZERNIKE_OP_CPU_H

#include "zernike_op.h"
#include <math.h>

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the Zernike op for CPUs
template <typename FT, typename CT>
class Zernike<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit Zernike(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_coords = context->input(0);
        const auto & in_coeffs = context->input(1);
        const auto & in_noll_index = context->input(2);
        const auto & in_pointing_error = context->input(3);
        const auto & in_antenna_scaling = context->input(4);
        const auto & in_parallactic_angle_sin = context->input(5);
        const auto & in_parallactic_angle_cos = context->input(6);

        // Extract Eigen tensors
        auto coords = in_coords.tensor<FT, 2>();
        auto coeffs = in_coeffs.tensor<CT, 4>();
        auto noll_index = in_noll_index.tensor<tensorflow::int32, 4>();
        auto pointing_error = in_pointing_error.tensor<FT, 4>();
        auto antenna_scaling = in_antenna_scaling.tensor<FT, 3>();
        auto parallactic_angle_sin = in_parallactic_angle_sin.tensor<FT, 2>();
        auto parallactic_angle_cos = in_parallactic_angle_cos.tensor<FT, 2>();

        int nsrc = in_coords.dim_size(0);
        int ntime = in_pointing_error.dim_size(0);
        int na = in_coeffs.dim_size(0);
        int nchan = in_coeffs.dim_size(1);
        int npoly = in_coeffs.dim_size(2);

        // Allocate output tensors
        // Allocate space for output tensor 'zernike_value'
        tf::Tensor * zernike_value_ptr = nullptr;
        tf::TensorShape zernike_value_shape = tf::TensorShape({ 
            nsrc, 
            ntime, 
            na, 
            nchan, 
            4 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, zernike_value_shape, &zernike_value_ptr));
        auto zernike_value = zernike_value_ptr->tensor<CT, 5>();

        #pragma omp parallel for
        for(int src = 0; src < nsrc; src++)
        {
            FT l = coords(src, 0);
            FT m = coords(src, 1);

            #pragma omp parallel for
            for(int time = 0; time < ntime; time++ )
            {
                #pragma omp parallel for
                for(int ant = 0; ant < na; ant++)
                {
                    FT pa_sin = parallactic_angle_sin(time, ant);
                    FT pa_cos = parallactic_angle_cos(time, ant);

                    FT l_error = l * pa_cos - m * pa_sin;
                    FT m_error = l * pa_sin + m * pa_cos;
                    #pragma omp parallel for
                    for(int chan = 0; chan < nchan; chan++)
                    {
                        l_error += pointing_error(time, ant, chan, 0);
                        m_error += pointing_error(time, ant, chan, 1);
                        l_error *= antenna_scaling(ant, chan, 0);
                        m_error *= antenna_scaling(ant, chan, 1);
                        l = l_error;
                        m = m_error;

                        FT rho = std::sqrt((l * l) + (m * m));
                        FT phi = std::atan2(l, m);

                        for(int corr = 0; corr < 4 ; corr++)
                        {
                            CT zernike_sum = 0;
                            for(int poly = 0; poly < npoly ; poly++)
                            {
                                zernike_sum += coeffs(ant, chan, poly, corr) * zernike(noll_index(ant, chan, poly, corr), rho, phi);
                            }
                            zernike_value(src, time, ant, chan, corr) = zernike_sum;
                        }
                    }
                }
            }
        }
    }

private:

    FT factorial(unsigned n){
        if(n == 0)
        { 
            return 1;
        }
        FT fac = 1;
        for(unsigned i = 1; i <= n; i++)
        {
            fac = fac * i;
        }
        return fac;
    }

    FT pre_fac(int k, int n, int m){
        FT numerator = factorial(n - k);
        if(k % 2 == 1) numerator *= -1;
        FT denominator = factorial(k) * factorial((n+m)/2.0 - k) * factorial((n-m)/2.0 - k);
        return numerator / denominator;
    }

    FT zernike_rad(int m, int n, FT rho){
        if(n < 0 || m < 0 || m > n){
            throw std::invalid_argument("m and n values are incorrect.");
        }
        FT radial_component = 0.0;
        for(int k = 0; k < ((n - m) / 2) + 1; k++)
        {
            radial_component += pre_fac(k,n,m) * pow(rho, n - 2.0 * k); 
        }
        return radial_component;
    }

    FT zernike(int j, FT rho, FT phi){
        if(rho > 1) 
        {
            return 0.;
        }
        // Convert from single-index Noll to regular double index
        int n = 0;
        j += 1;
        int j1 = j - 1;
        while(j1 > n)
        {
            n += 1;
            j1 -= n;
        }
        int m = ((n%2) + 2 * ((j1 + ((n+1)%2)) / 2));
        if(j % 2 == 1) m *= -1;
        // Get Zernike polynomials
        if(m > 0)
        {
            return zernike_rad(m, n, rho) * cos(m * phi);

        }
        else if(m < 0)
        {
            return zernike_rad(-1 * m, n, rho) * sin(-1 * m * phi);
        }
        return zernike_rad(0, n, rho);
    }
};

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_CPU_H