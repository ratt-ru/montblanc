import os.path
import numpy as np
from BaseSharedData import *

from pyrap.tables import table

class MeasurementSetSharedData(GPUSharedData):
    ANTENNA_TABLE = "ANTENNA"
    SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

    def __init__(self, ms_file, nsrc, dtype=np.float64):
        # Do some checks on the supplied filename
        if not isinstance(ms_file, str):
            raise TypeError, 'ms_file is not a string'

        if not os.path.isdir(ms_file):
            raise ValueError, '%s does not appear to be a valid measurement set' % ms_file

        # Store the measurement set filename
        self.ms_file = ms_file

        # Open the measurement set
        t = table(self.ms_file, ack=False)

        # Get the UVW coordinates
        uvw=t.getcol("UVW").T.astype(dtype)

        # Open the antenna table
        ta=table(self.ms_file + os.sep + MeasurementSetSharedData.ANTENNA_TABLE, ack=False)
        # Open the spectral window table
        tf=table(self.ms_file + os.sep + MeasurementSetSharedData.SPECTRAL_WINDOW, ack=False)
        f=tf.getcol("CHAN_FREQ").astype(dtype)

        na = len(ta.getcol("NAME"))
        nbl = (na*(na-1))/2
        nchan = f.size
        ntime = uvw.shape[1] / nbl

        super(MeasurementSetSharedData, self).__init__(\
            na=na,nchan=nchan,ntime=ntime,nsrc=nsrc,dtype=dtype)

        self.wavelength = 3e8/f

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        self.keys = (np.arange(np.product(self.jones_shape[:-1]))
            *self.jones_shape[-1]).astype(np.int32)
        self.keys_gpu = gpuarray.to_gpu(self.keys)

        self.uvw = uvw.reshape(self.uvw_shape).copy()

		# Transfer the uvw coordinates 
        # and antenna pairs to the GPU
        self.transfer_uvw(uvw)
        self.transfer_ant_pairs(self.get_default_ant_pairs())
        self.transfer_wavelength(self.wavelength)

        # TODO: Setting the reference wavelength to a frequency makes no sense,
        # but this matches Cyril's predict...
        # First dimension also seems to be of size 1 here...
        self.set_refwave(f[0][nchan/2])

        t.close()
        ta.close()
        tf.close()