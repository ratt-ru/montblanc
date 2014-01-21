from pycuda.compiler import SourceModule

serial_predict = self.mod = SourceModule("""
#include <pycuda-complex.hpp>
#include \"math_constants.h\"

__device__ void Product2by2(
    const pycuda::complex<double> * lhs,
    const pycuda::complex<double> * rhs,
    pycuda::complex<double> * result)
{
    const pycuda::complex<double> & a00 = lhs[0];
    const pycuda::complex<double> & a10 = lhs[2];
    const pycuda::complex<double> & a01 = lhs[1];
    const pycuda::complex<double> & a11 = lhs[3];

    const pycuda::complex<double> & b00 = rhs[0];
    const pycuda::complex<double> & b10 = rhs[2];
    const pycuda::complex<double> & b01 = rhs[1];
    const pycuda::complex<double> & b11 = rhs[3];

    result[0]=a00*b00+a01*b10;
    result[1]=a00*b01+a01*b11;
    result[2]=a10*b00+a11*b10;
    result[3]=a10*b01+a11*b11;
}

__device__ void Product2by2H(
    const pycuda::complex<double> * lhs,
    const pycuda::complex<double> * rhs,
    pycuda::complex<double> * result)
{
    const pycuda::complex<double> & a00 = lhs[0];
    const pycuda::complex<double> & a10 = lhs[2];
    const pycuda::complex<double> & a01 = lhs[1];
    const pycuda::complex<double> & a11 = lhs[3];

    const pycuda::complex<double> b00 = pycuda::conj(rhs[0]);
    const pycuda::complex<double> b10 = pycuda::conj(rhs[2]);
    const pycuda::complex<double> b01 = pycuda::conj(rhs[1]);
    const pycuda::complex<double> b11 = pycuda::conj(rhs[3]);

    result[0]=a00*b00+a01*b10;
    result[1]=a00*b01+a01*b11;
    result[2]=a10*b00+a11*b10;
    result[3]=a10*b01+a11*b11;
}


__global__ void predict(
    pycuda::complex<double> * VisIn,
    double * UVWin,
    double * LM,
    long * A0,
    long * A1,
    double * wavelength,
    pycuda::complex<double> * solution,
    int ndir, int nchan, int na, int nrows)
{
//    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    const pycuda::complex<double> I = pycuda::complex<double>(0,1);
    const pycuda::complex<double> c0 = 2.0*CUDART_PI_F*I;

    const double refwave = 1e6;

    pycuda::complex<double> * p0 = VisIn;
    double * p1 = UVWin;

    return;

    for(int dd=0;dd<ndir;dd++)
    {
        double l=LM[0];
        double m=LM[1];
        double fI=LM[2]; // flux
        double alpha=LM[3];
        double fQ=LM[4];
        double fU=LM[5];
        double fV=LM[6];        
        double n = sqrt(1.-l*l-m*m)-1.;

        LM+=6;
        VisIn = p0;
        UVWin = p1;

        pycuda::complex<double> sky[4] =
        {
            fI+fQ,
            fU+I*fV,
            fU-I*fV,
            fI-fQ
        };

        double phase = UVWin[0]*l + UVWin[1]*m + UVWin[2]*n;
        UVWin += 3;
    #if 1

        for(int bl=0; bl<nrows; ++bl)
        {
            for(int ch=0; ch<nchan; ++ch)
            { 
                pycuda::complex<double> c1=c0/wavelength[ch];
                double this_flux=pow(refwave/wavelength[ch],alpha);
                pycuda::complex<double> result=this_flux*pycuda::exp(phase*c1);

                pycuda::complex<double> JJ[4];
                //Sols shape: Nd, Nf, Na, 4
                pycuda::complex<double> * J0
                    = solution + dd*na*nchan*4 + ch*na*4 + A0[bl]*4;
                pycuda::complex<double> * J1
                    = solution + dd*na*nchan*4 + ch*na*4 + A1[bl]*4;

                Product2by2(J0, J1, JJ);

                VisIn[0] += JJ[0]*result;
                VisIn[1] += JJ[1]*result;
                VisIn[2] += JJ[2]*result;
                VisIn[3] += JJ[3]*result;

                VisIn += 4;
            }
        }

    #endif
    }    
}
""")
