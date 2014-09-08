import os.path
import numpy as np
import pycuda.gpuarray as gpuarray

from pyrap.tables import table

import montblanc
import montblanc.BaseSharedData

from montblanc.BaseSharedData import get_nr_of_baselines
from montblanc.impl.biro.v1.BiroSharedData import BiroSharedData

class MeasurementSetSharedData(BiroSharedData):
    ANTENNA_TABLE = "ANTENNA"
    SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

    def __init__(self, ms_file, npsrc, ngsrc, dtype=np.float64, **kwargs):
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

        # Open the antenna table
        ant_path = os.path.join(self.ms_file, MeasurementSetSharedData.ANTENNA_TABLE)
        ta=table(ant_path, ack=False)

        # Open the spectral window table
        freq_path = os.path.join(self.ms_file, MeasurementSetSharedData.SPECTRAL_WINDOW)
        tf=table(freq_path, ack=False)
        f=tf.getcol("CHAN_FREQ").astype(dtype)

        # Determine the problem dimensions
        na = ta.nrows()
        nbl = get_nr_of_baselines(na)
        nchan = f.size
        ntime = t.nrows() // nbl

        super(MeasurementSetSharedData, self).__init__(\
            na=na,nchan=nchan,ntime=ntime,npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)

        # Check that nbl agrees with version from constructor
        assert nbl == self.nbl

        # Check that we're getting the correct shape...
        expected_uvw_shape = (3, ntime*nbl)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = t.getcol('UVW')
        assert ms_uvw.shape == (ntime*nbl, 3), \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape,expected_uvw_shape)
        uvw = ms_uvw.reshape(ntime, nbl, 3).transpose(2,1,0) \
            .astype(self.ft)
        self.transfer_uvw(np.ascontiguousarray(uvw))

        # Determine the wavelengths
        wavelength = (montblanc.constants.C/f[0]).astype(self.ft)
        self.transfer_wavelength(wavelength)

        # First dimension also seems to be of size 1 here...
        # Divide speed of light by frequency to get the wavelength here.
        self.set_ref_wave(montblanc.constants.C/tf.getcol('REF_FREQUENCY')[0])

        # Get the baseline antenna pairs and correct the axes
        ant1 = t.getcol('ANTENNA1').reshape(ntime,nbl).transpose(1,0)
        ant2 = t.getcol('ANTENNA2').reshape(ntime,nbl).transpose(1,0)

        expected_ant_shape = (nbl,ntime)
        
        assert expected_ant_shape == ant1.shape, \
            'ANTENNA1 shape is %s != expected %s' % (ant1.shape,expected_ant_shape)

        assert expected_ant_shape == ant2.shape, \
            'ANTENNA2 shape is %s != expected %s' % (ant2.shape,expected_ant_shape)

        ant_pairs = np.vstack((ant1,ant2)).reshape(self.ant_pairs_shape)

        # Transfer the uvw coordinates, antenna pairs and wavelengths to the GPU
        self.transfer_ant_pairs(np.ascontiguousarray(ant_pairs))

        # Load in visibility data, if it exists.
        if t.colnames().count('DATA') > 0:
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = t.getcol('DATA').reshape(ntime,nbl,nchan,4) \
                .transpose(3,1,2,0).astype(self.ct)
            self.transfer_bayes_data(np.ascontiguousarray(vis_data))

        # Should we initialise our weights from the MS data?
        init_weights = kwargs.get('init_weights', None)
        valid = [None, 'sigma', 'weight']

        if not init_weights in valid:
            raise ValueError, 'init_weights should be set to None, ''sigma'' or ''weights'''

        # Load in flag and weighting data, if it exists
        if init_weights is not None and t.colnames().count('FLAG') > 0:
            # Get the flag data. Data flagged as True should be ignored,
            # therefore we flip the values so that when we multiply flag
            # by the weights later, a False value zeroes out the weight
            flag = (t.getcol('FLAG') == False)
            flag_row = (t.getcol('FLAG_ROW') == False)

            # Incorporate the flag_row data into the larger
            # flag matrix
            flag = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

            if init_weights == 'weight':
                # Obtain weighting information from WEIGHT_SPECTRUM
                # preferably, otherwise WEIGHT.
                if t.colnames().count('WEIGHT_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*t.getcol('WEIGHT_SPECTRUM')
                elif t.colnames().count('WEIGHT') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHT.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (t.getcol('WEIGHT')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(self.ft)
            elif init_weights == 'sigma':
                # Obtain weighting information from SIGMA_SPECTRUM
                # preferably, otherwise SIGMA.
                if t.colnames().count('SIGMA_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*t.getcol('SIGMA_SPECTRUM')
                elif t.colnames().count('SIGMA') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHT.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (t.getcol('SIGMA')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(self.ft)
            else:
                raise Exception, 'init_weights used incorrectly!'

            assert weight_vector.shape == (ntime*nbl, nchan, 4)

            weight_vector = weight_vector.reshape(ntime,nbl,nchan,4) \
                .transpose(3,1,2,0).astype(self.ft)

            self.transfer_weight_vector(np.ascontiguousarray(weight_vector))

        t.close()
        ta.close()
        tf.close()