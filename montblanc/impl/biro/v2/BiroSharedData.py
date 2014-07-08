import numpy as np

from montblanc.BaseSharedData import BaseSharedData
from montblanc.BaseSharedData import DEFAULT_NA
from montblanc.BaseSharedData import DEFAULT_NCHAN
from montblanc.BaseSharedData import DEFAULT_NTIME
from montblanc.BaseSharedData import DEFAULT_NPSRC
from montblanc.BaseSharedData import DEFAULT_NGSRC
from montblanc.BaseSharedData import DEFAULT_DTYPE

class BiroSharedData(BaseSharedData):
    """ Shared Data implementation for BIRO """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, dtype=DEFAULT_DTYPE, **kwargs):
        """
        BiroSharedData Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
        Keyword Arguments:
            device : pycuda.device.Device
                CUDA device to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSharedData object.
        """
        super(BiroSharedData, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, dtype=dtype, **kwargs)

        sd = self
        na, nbl, nchan, ntime = sd.na, sd.nbl, sd.nchan, sd.ntime
        npsrc, ngsrc, nsrc = sd.npsrc, sd.ngsrc, sd.nsrc
        ft, ct = sd.ft, sd.ct

        # Curry the register_array function for simplicity
        def reg(name,shape,dtype):
            self.register_array(name=name,shape=shape,dtype=dtype,
                registrant='BaseSharedData', gpu=True, cpu=False,
                shape_member=True, dtype_member=True)

        def reg_prop(name,dtype,default):
            self.register_property(name=name,dtype=dtype,
                default=default,registrant='BaseSharedData', setter=True)

        # Set up gaussian scaling parameters
        # Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
        # and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
        fwhm2int = 1.0/np.sqrt(np.log(256))
        # Note that we don't divide by speed of light here. meqtrees code operates
        # on frequency, while we're dealing with wavelengths.
        reg_prop('gauss_scale', ft, fwhm2int*np.sqrt(2)*np.pi)
        reg_prop('ref_wave', ft, 0.0)

        reg_prop('sigma_sqrd', ft, 1.0)
        reg_prop('X2', ft, 0.0)
        reg_prop('beam_width', ft, 65)
        reg_prop('beam_clip', ft, 1.0881)

        reg(name='uvw', shape=(3,na,ntime), dtype=ft)
        reg(name='ant_pairs', shape=(2,nbl,ntime), dtype=np.int32)

        reg(name='lm', shape=(2,nsrc), dtype=ft)
        reg(name='brightness', shape=(5,ntime,nsrc), dtype=ft)
        reg(name='gauss_shape', shape=(3, ngsrc), dtype=ft)

        reg(name='wavelength', shape=(nchan,), dtype=ft)
        reg(name='point_errors', shape=(2,na,ntime), dtype=ft)
        reg(name='weight_vector', shape=(4,nbl,ntime,nchan), dtype=ft)
        reg(name='bayes_data', shape=(4,nbl,ntime,nchan), dtype=ct)

        jones_scalar_shape = (na,ntime,nsrc,nchan)

        reg(name='jones_scalar', shape=jones_scalar_shape, dtype=ct)
        reg(name='vis', shape=(4,nbl,ntime,nchan), dtype=ct)
        reg(name='chi_sqrd_result', shape=(nbl,ntime,nchan), dtype=ft)