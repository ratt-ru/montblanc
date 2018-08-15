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
        

        // Extract Eigen tensors
        auto coords = in_coords.tensor<FT, 5>();
        auto coeffs = in_coeffs.tensor<CT, 3>();
        auto noll_index = in_noll_index.tensor<FT, 3>();

        int nsrc = in_coords.dim_size(1);
        int ntime = in_coords.dim_size(2);
        int na = in_coords.dim_size(3);
        int nchan = in_coords.dim_size(4);
        int npoly = in_coeffs.dim_size(2);
        

        // Allocate output tensors
        // Allocate space for output tensor 'zernike_value'
        tf::Tensor * zernike_value_ptr = nullptr;
        tf::TensorShape zernike_value_shape = tf::TensorShape({ 
            nsrc, 
            ntime, 
            na, 
            nchan });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, zernike_value_shape, &zernike_value_ptr));
        // Create output tensor
        auto zernike_value = zernike_value_ptr->tensor<CT, 4>();

        for(int src = 0; src < nsrc ; src++){
            for(int time = 0; time < ntime; time++){
                for(int ant = 0; ant < na; ant++){
                    for(int chan = 0; chan < nchan; chan++){
                        // Get (l,m,freq) coordinates
                        FT l = coords(0, src, time, ant, chan);
                        FT m = coords(1, src, time, ant, chan);
                        FT freq = coords(2, src, time, ant, chan);

                        // Convert from (l, m) coordinates to polar coordinates
                        FT rho = sqrt((l * l)+(m * m));
                        FT phi = atan2(l, m);
                        
                        CT zernike_sum = 0;
                        for(int poly = 0; poly < npoly; poly++){
                            zernike_sum+= coeffs(ant, chan, poly) * zernike(noll_index(ant, chan, poly), rho, phi);
                        }
                        zernike_value(src, time, ant, chan) = zernike_sum;
                    }
                }
            }
        }
        
    }


private:

    FT factorial(unsigned n){
        if(n == 0) return 1;
        FT fac = 1;
        for(int i = 1; i <= n; i++){
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
        for(int k = 0; k < ((n - m) / 2) + 1; k++){
            radial_component += pre_fac(k,n,m) * pow(rho, n - 2.0 * k);  
           }
        return radial_component;
    }

    CT zernike(int j, FT rho, FT phi){
        if(rho > 1) return 0;

        // Convert from single-index Noll to regular double index
        int n = 0;
        j += 1;
        int j1 = j - 1;
        while(j1 > n){
            n += 1;
            j1 -= n;
        }
        int m = ((n%2) + 2 * ((j1 + ((n+1)%2)) / 2));
        if(j % 2 == 1) m *= -1;

        // Get Zernike polynomials
        if(m > 0){
            return zernike_rad(m, n, rho) * cos(m * phi);
        }
        else if(m < 0){
            return zernike_rad(-1 * m, n, rho) * sin(-1 * m * phi);
        }
        return zernike_rad(0, n, rho);
    }
};

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_CPU_H