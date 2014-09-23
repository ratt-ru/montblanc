import os.path
import numpy as np
import pycuda.gpuarray as gpuarray

from pyrap.tables import table

import montblanc
import montblanc.BaseSolver

from montblanc.impl.biro.v2.BiroSolver import BiroSolver

class MeasurementSetSolver(BiroSolver):
    ANTENNA_TABLE = "ANTENNA"
    SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

    def __init__(self, msfile, npsrc, ngsrc, dtype=np.float64, **kwargs):
        # Do some checks on the supplied filename
        if not isinstance(msfile, str):
            raise TypeError, 'msfile is not a string'

        if not os.path.isdir(msfile):
            raise ValueError, '%s does not appear to be a valid measurement set' % msfile

        # Store the measurement set filename
        self.msfile = msfile

        # Open the measurement set
        t = table(self.msfile, ack=False).query('ANTENNA1 != ANTENNA2')

        # Open the antenna table
        ant_path = os.path.join(self.msfile, MeasurementSetSolver.ANTENNA_TABLE)
        ta=table(ant_path, ack=False)

        # Open the spectral window table
        freq_path = os.path.join(self.msfile, MeasurementSetSolver.SPECTRAL_WINDOW)
        tf=table(freq_path, ack=False)
        f=tf.getcol("CHAN_FREQ").astype(dtype)

        # Turn on auto_correlations
        kwargs['auto_correlations'] = False

        # Determine the problem dimensions
        na = ta.nrows()
        nbl = montblanc.nr_of_baselines(na, kwargs['auto_correlations'])
        nchan = f.size
        ntime = t.nrows() // nbl

        super(MeasurementSetSolver, self).__init__(\
            na=na,nchan=nchan,ntime=ntime,npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)

        # Check that nbl agrees with version from constructor
        assert nbl == self.nbl

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = t.getcol('UVW')
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape,expected_uvw_shape)
        uvw_rec = self.get_array_record('uvw')
        uvw=np.empty(shape=uvw_rec.shape, dtype=uvw_rec.dtype)
        uvw[:,:,1:na] = ms_uvw.reshape(ntime, nbl, 3).transpose(2,0,1) \
            .astype(self.ft)[:,:,:na-1]
        uvw[:,:,0] = self.ft(0)
        self.transfer_uvw(np.ascontiguousarray(uvw))

        # Determine the wavelengths
        wavelength = (montblanc.constants.C/f[0]).astype(self.ft)
        self.transfer_wavelength(wavelength)

        # First dimension also seems to be of size 1 here...
        # Divide speed of light by frequency to get the wavelength here.
        self.set_ref_wave(montblanc.constants.C/tf.getcol('REF_FREQUENCY')[0])

        # Get the baseline antenna pairs and correct the axes
        ant1 = t.getcol('ANTENNA1').reshape(ntime,nbl)
        ant2 = t.getcol('ANTENNA2').reshape(ntime,nbl)

        expected_ant_shape = (ntime,nbl)
        
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
                .transpose(3,0,1,2).astype(self.ct)
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
                .transpose(3,0,1,2).astype(self.ft)

            self.transfer_weight_vector(np.ascontiguousarray(weight_vector))

        #self.test_uvw_relations(ms_uvw, ant_pairs)

        t.close()
        ta.close()
        tf.close()

    def test_uvw_relations(self, msuvw, ant_pairs):
        """ Test that the uvw relation holds """
        ap = ant_pairs.reshape(2,self.ntime*self.nbl)

        # Create 1D indices from the flattened antenna pair array.
        # Multiply it by size constant and add tiling corresponding
        # to the indexing.
        ant0 = ap[0] + np.repeat(np.arange(self.ntime),self.nbl)*self.na
        ant1 = ap[1] + np.repeat(np.arange(self.ntime),self.nbl)*self.na

        uvw_rec = self.get_array_record('uvw')
        uvw=np.empty(shape=uvw_rec.shape, dtype=uvw_rec.dtype)
        uvw[:,:,1:self.na] = msuvw.reshape(self.ntime, self.nbl, 3).transpose(2,0,1) \
            .astype(self.ft)[:,:,:self.na-1]
        uvw[:,:,0] = self.ft(0)

        print 'uvw', uvw
        print 'uvw diff', uvw[:,:,:] - uvw[:,:,:]

        """
        ap = ant_pairs.reshape(2,self.ntime*self.nbl)

        # Create 1D indices from the flattened antenna pair array.
        # Multiply it by size constant and add tiling corresponding
        # to the indexing.
        ant0 = ap[0] + np.repeat(np.arange(self.ntime),self.nbl)*self.na
        ant1 = ap[1] + np.repeat(np.arange(self.ntime),self.nbl)*self.na

        print uvw.shape
        print 'ant0 == ant1 %s' % (ant0 == ant1).all()


        np.savetxt('file.txt',(uvw[:,:] - (uvw[ant1,:] - uvw[ant0,:])))

        idx = np.empty(shape=(self.ntime*self.nbl),dtype=np.int32)

        suvw = uvw.reshape(self.ntime,self.nbl,3)

        # TODO. Inefficient indexing. Figure something out based
        # on RimeCPU.compute_gaussian_shape
        for t in range(self.ntime):
            for bl in range(self.nbl):
                #idx[t*self.nbl + bl] = t*self.na + ant_pairs[0,t,bl]
                assert np.allclose(
                    suvw[t,bl,:],
                    suvw[t,ant_pairs[1,t,bl],:] - suvw[t,ant_pairs[0,t,bl],:]), \
                    'UVW Relation does not hold for timestep %d baseline %d!' % (t,bl)
        """
