import numpy as np

import montblanc
import montblanc.impl.common.loaders

class MeasurementSetLoader(montblanc.impl.common.loaders.MeasurementSetLoader):
    def load(self, solver, **kwargs):
        """
        Load the Measurement Set
        """
        tm = self.tables['main']
        ta = self.tables['ant']
        tf = self.tables['freq']

        na, nbl, ntime, nchan = solver.na, solver.nbl, solver.ntime, solver.nchan

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = tm.getcol('UVW')
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape,uvw_shape)
        uvw = ms_uvw.reshape(ntime,nbl,3).transpose(2,1,0) \
            .astype(solver.ft)
        solver.transfer_uvw(np.ascontiguousarray(uvw))

        # Determine the wavelengths
        freqs = tf.getcol('CHAN_FREQ')
        wavelength = (montblanc.constants.C/freqs[0]).astype(solver.ft)
        solver.transfer_wavelength(wavelength)

        # First dimension also seems to be of size 1 here...
        # Divide speed of light by frequency to get the wavelength here.
        ref_freq = tf.getcol('REF_FREQUENCY')[0]
        ref_wave = montblanc.constants.C/ref_freq
        solver.set_ref_wave(ref_wave)

        # Get the baseline antenna pairs and correct the axes
        ant1 = tm.getcol('ANTENNA1').reshape(ntime,nbl).transpose(1,0)
        ant2 = tm.getcol('ANTENNA2').reshape(ntime,nbl).transpose(1,0)

        expected_ant_shape = (nbl,ntime)
        
        assert expected_ant_shape == ant1.shape, \
            'ANTENNA1 shape is %s != expected %s' % (ant1.shape,expected_ant_shape)

        assert expected_ant_shape == ant2.shape, \
            'ANTENNA2 shape is %s != expected %s' % (ant2.shape,expected_ant_shape)

        ant_pairs = np.vstack((ant1,ant2)).reshape(solver.ant_pairs_shape)

        # Transfer the uvw coordinates, antenna pairs and wavelengths to the GPU
        solver.transfer_ant_pairs(np.ascontiguousarray(ant_pairs))

        # Load in visibility data, if it exists.
        if tm.colnames().count('DATA') > 0:
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = tm.getcol('DATA').reshape(ntime,nbl,nchan,4) \
                .transpose(3,1,2,0).astype(solver.ct)
            solver.transfer_bayes_data(np.ascontiguousarray(vis_data))

        # Should we initialise our weights from the MS data?
        init_weights = kwargs.get('init_weights', None)
        valid = [None, 'sigma', 'weight']

        if not init_weights in valid:
            raise ValueError, 'init_weights should be set to None, ''sigma'' or ''weights'''

        # Load in flag and weighting data, if it exists
        if init_weights is not None and tm.colnames().count('FLAG') > 0:
            # Get the flag data. Data flagged as True should be ignored,
            # therefore we flip the values so that when we multiply flag
            # by the weights later, a False value zeroes out the weight
            flag = (tm.getcol('FLAG') == False)
            flag_row = (tm.getcol('FLAG_ROW') == False)

            # Incorporate the flag_row data into the larger
            # flag matrix
            flag = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

            if init_weights == 'weight':
                # Obtain weighting information from WEIGHT_SPECTRUM
                # preferably, otherwise WEIGHtm.
                if tm.colnames().count('WEIGHT_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM firstm.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('WEIGHT_SPECTRUM')
                elif tm.colnames().count('WEIGHT') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHtm.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (tm.getcol('WEIGHT')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(solver.ft)
            elif init_weights == 'sigma':
                # Obtain weighting information from SIGMA_SPECTRUM
                # preferably, otherwise SIGMA.
                if tm.colnames().count('SIGMA_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM firstm.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('SIGMA_SPECTRUM')
                elif tm.colnames().count('SIGMA') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHtm.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (tm.getcol('SIGMA')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(solver.ft)
            else:
                raise Exception, 'init_weights used incorrectly!'

            assert weight_vector.shape == (ntime*nbl, nchan, 4)

            weight_vector = weight_vector.reshape(ntime,nbl,nchan,4) \
                .transpose(3,1,2,0).astype(solver.ft)

            solver.transfer_weight_vector(np.ascontiguousarray(weight_vector))

    def __enter__(self):
        return super(MeasurementSetLoader,self).__enter__()

    def __exit__(self, type, value, traceback):
        return super(MeasurementSetLoader,self).__exit__(type,value,traceback)