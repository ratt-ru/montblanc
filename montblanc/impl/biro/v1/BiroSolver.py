import numpy as np

from montblanc.BaseSolver import BaseSolver
from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

class BiroSolver(BaseSolver):
    """ Shared Data implementation for BIRO """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, dtype=DEFAULT_DTYPE,
        pipeline=None, **kwargs):
        """
        BiroSolver Constructor

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
            pipeline : list of nodes
                nodes defining the GPU kernels used to solve this RIME
        Keyword Arguments:
            context : pycuda.driver.Context
                CUDA context to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSolver object.
        """

        # Turn off auto_correlations
        kwargs['auto_correlations'] = False

        super(BiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, dtype=dtype, pipeline=pipeline, **kwargs)

        slvr = self
        na, nbl, nchan, ntime = slvr.na, slvr.nbl, slvr.nchan, slvr.ntime
        npsrc, ngsrc, nsrc = slvr.npsrc, slvr.ngsrc, slvr.nsrc
        ft, ct = slvr.ft, slvr.ct

        # Curry the register_array function for simplicity
        def reg(name,shape,dtype):
            self.register_array(name=name,shape=shape,dtype=dtype,
                registrant='BaseSolver', gpu=True, cpu=False,
                shape_member=True, dtype_member=True)

        def reg_prop(name,dtype,default):
            self.register_property(name=name,dtype=dtype,
                default=default,registrant='BaseSolver', setter=True)

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

        reg(name='uvw', shape=(3,'nbl','ntime'), dtype=ft)
        reg(name='ant_pairs', shape=(2,'nbl','ntime'), dtype=np.int32)

        reg(name='lm', shape=(2,'nsrc'), dtype=ft)
        reg(name='brightness', shape=(5,'ntime','nsrc'), dtype=ft)
        reg(name='gauss_shape', shape=(3, 'ngsrc'), dtype=ft)

        reg(name='wavelength', shape=('nchan',), dtype=ft)
        reg(name='point_errors', shape=(2,'na','ntime'), dtype=ft)
        reg(name='weight_vector', shape=(4,'nbl','nchan','ntime'), dtype=ft)
        reg(name='bayes_data', shape=(4,'nbl','nchan','ntime'), dtype=ct)

        reg(name='jones', shape=(4,'nbl','nchan','ntime','nsrc'), dtype=ct)        
        reg(name='vis', shape=(4,'nbl','nchan','ntime'), dtype=ct)
        reg(name='chi_sqrd_result', shape=('nbl','nchan','ntime'), dtype=ft)

        # Get the numeric jones shape, so that we can calculate the key array size
        njones_shape = self.get_array_record('jones').shape

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        keys = (np.arange(np.product(njones_shape[:-1]))
            *njones_shape[-1]).astype(np.int32)

        reg(name='keys', shape=keys.shape, dtype=np.int32)

        slvr.transfer_keys(keys)

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, nbl, ntime), dtype=np.int32]) containing the
        default antenna pairs for each baseline at each timestep.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna. 
        slvr = self

        return np.repeat(np.int32(np.triu_indices(slvr.na,1)),
            slvr.ntime).reshape(2,slvr.nbl,slvr.ntime)
