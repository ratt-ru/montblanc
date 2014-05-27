import os.path
import numpy as np
import pycuda.gpuarray as gpuarray

from pyrap.tables import table

import montblanc.BaseSharedData as BaseSharedData
from montblanc.BaseSharedData import GPUSharedData

class MeasurementSetSharedData(GPUSharedData):
    ANTENNA_TABLE = "ANTENNA"
    SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

    def __init__(self, ms_file, nsrc, dtype=np.float64,device=None):
        # Do some checks on the supplied filename
        if not isinstance(ms_file, str):
            raise TypeError, 'ms_file is not a string'

        if not os.path.isdir(ms_file):
            raise ValueError, '%s does not appear to be a valid measurement set' % ms_file

        # Store the measurement set filename
        self.ms_file = ms_file

        # Open the measurement set
        ms = table(self.ms_file, ack=False)
        # Strip out all auto-correlated baselines
        t = ms.query('ANTENNA1 != ANTENNA2')

        # Get the UVW coordinates
        uvw=t.getcol("UVW").T.astype(dtype)

        ant_path = os.path.join(self.ms_file, MeasurementSetSharedData.ANTENNA_TABLE)
        freq_path = os.path.join(self.ms_file, MeasurementSetSharedData.SPECTRAL_WINDOW)
        # Open the antenna table
        ta=table(ant_path, ack=False)
        # Open the spectral window table
        tf=table(freq_path, ack=False)
        f=tf.getcol("CHAN_FREQ").astype(dtype)

        # Determine the problem dimensions
        na = len(ta.getcol("NAME"))
        nbl = BaseSharedData.get_nr_of_baselines(na)
        nchan = f.size
        ntime = uvw.shape[1] // nbl

        # Check that we're getting the correct shape...
        expected_uvw_shape = (3, nbl*ntime)

        if expected_uvw_shape != uvw.shape:
            raise ValueError, 'uvw.shape %s != expected %s' % (uvw.shape,expected_uvw_shape)

        super(MeasurementSetSharedData, self).__init__(\
            na=na,nchan=nchan,ntime=ntime,nsrc=nsrc,dtype=dtype)

        # Check that nbl agrees with version from constructor
        assert nbl == self.nbl

        # Reshape the flatten array
        uvw = uvw.reshape(self.uvw_shape).copy()

        # Determine the wavelengths
        wavelength = 3e8/f[0]

        # Get the baseline antenna pairs
        ant1 = t.getcol('ANTENNA1')
        ant2 = t.getcol('ANTENNA2')

        # Load in visibility data, if it exists.
        if t.colnames().count('DATA') > 0:
            # Obtain visibilities stored in the DATA column
            # This comes in as (nbl*ntime,nchan,4)
            vis_data = t.getcol('DATA')

            # Shift the dim 4 axis to the front,
            # reshape to separate nbl and ntime, and then
            # swap channel and time
            vis_data=np.rollaxis(
                np.rollaxis(vis_data,axis=2,start=0).reshape(
                    4,nbl,ntime,nchan),
                axis=3,start=2).astype(self.ct).copy()
            self.transfer_bayes_data(vis_data)

        expected_ant_shape = (nbl*ntime,)
        
        if expected_ant_shape != ant1.shape:
            raise ValueError, 'ANTENNA1 shape is %s != expected %s' % (ant1.shape,expected_ant_shape)

        if expected_ant_shape != ant2.shape:
            raise ValueError, 'ANTENNA2 shape is %s != expected %s' % (ant2.shape,expected_ant_shape)

        ant_pairs = np.vstack((ant1,ant2)).reshape(self.ant_pairs_shape)

        # Transfer the uvw coordinates, antenna pairs and wavelengths to the GPU
        self.transfer_uvw(uvw)
        self.transfer_ant_pairs(ant_pairs)
        self.transfer_wavelength(wavelength)

        # First dimension also seems to be of size 1 here...
        self.set_ref_freq(tf.getcol('REF_FREQUENCY')[0])

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        self.keys = (np.arange(np.product(self.jones_shape[:-1]))
            *self.jones_shape[-1]).astype(np.int32)
        self.keys_gpu = gpuarray.to_gpu(self.keys)

        t.close()
        ta.close()
        tf.close()
