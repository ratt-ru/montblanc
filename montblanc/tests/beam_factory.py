from os.path import exists as pexists
from os.path import join as pjoin

from astropy.io import fits
import numpy as np

from montblanc.impl.rime.tensorflow.sources.fits_beam_source_provider import (
                            REIM, LINEAR_CORRELATIONS, CIRCULAR_CORRELATIONS,
                            FitsFilenameTemplate, _create_filenames)

BITPIX_MAP = {
    np.int8: 8,
    np.uint8: 8,
    np.int16: 16,
    np.uint16: 16,
    np.int32: 32,
    np.uint32: 32,
    np.float32: -32,
    np.float64: -64,
}

DEFAULT_SCHEMA = pjoin("test_beam_$(corr)_$(reim).fits")

def beam_factory(polarisation_type='linear',
                    frequency=None,
                    dtype=np.float64,
                    schema=DEFAULT_SCHEMA,
                    overwrite=False):
    """ Generate a MeqTrees compliant beam cube """

    # MeerKAT l-band, 64 channels
    if frequency is None:
        frequency = np.linspace(.856e9, .856e9*2, 64,
                                endpoint=True, dtype=np.float64)
    # Generate a linear space of grid frequencies
    gfrequency = np.linspace(frequency[0], frequency[-1],
                            32, dtype=np.float64)
    bandwidth = gfrequency[-1] - frequency[0]
    bandwidth_delta = bandwidth / gfrequency.shape[0]-1

    if polarisation_type == 'linear':
        CORR = LINEAR_CORRELATIONS
    elif polarisation_type == 'circular':
        CORR = CIRCULAR_CORRELATIONS
    else:
        raise ValueError("Invalid polarisation_type %s" % polarisation_type)

    # List of key values of the form:
    #
    #    (key, None)
    #    (key, (value,))
    #    (key, (value, comment))
    #
    # We put them in a list so that they are added to the
    # FITS header in the correct order
    axis1 = [
        ("CTYPE", ('X', "points right on the sky")),
        ("CUNIT", ('DEG', 'degrees')),
        ("NAXIS", (257, "number of X")),
        ("CRPIX", (129, "reference pixel (one relative)")),
        ("CRVAL", (0.0110828777007, "degrees")),
        ("CDELT", (0.011082, "degrees"))]

    axis2 = [
        ("CTYPE", ('Y', "points up on the sky")),
        ("CUNIT", ('DEG', 'degrees')),
        ("NAXIS", (257, "number of Y")),
        ("CRPIX", (129, "reference pixel (one relative)")),
        ("CRVAL", (-2.14349358381E-07, "degrees")),
        ("CDELT", (0.011082, "degrees"))]

    axis3 = [
        ("CTYPE", ('FREQ', )),
        ("CUNIT", None),
        ("NAXIS", (gfrequency.shape[0], "number of FREQ")),
        ("CRPIX", (1, "reference frequency position")),
        ("CRVAL", (gfrequency[0], "reference frequency")),
        ("CDELT", (bandwidth_delta, "frequency step in Hz"))]

    axis4 = [
        ("CTYPE", ('STOKES', )),
        ("CUNIT", None),
        ("NAXIS", (1, "number of STOKES")),
        ("CRPIX", (1, "reference stokes index")),
        ("CRVAL", (1, "first stokes index")),
        ("CDELT", (-5,))]

    axes = [axis1, axis2, axis3]

    metadata = [
        ('SIMPLE', True),
        ('BITPIX', BITPIX_MAP[dtype]),
        ('NAXIS', len(axes)),
        ('OBSERVER', "Astronomer McAstronomerFace"),
        ('ORIGIN', "Artificial"),
        ('TELESCOP', "Telescope"),
        ('OBJECT', 'beam'),
        ('EQUINOX', 2000.0),
    ]

    # Create header and set metadata
    header = fits.Header()
    header.update(metadata)

    # Now set the key value entries for each axis
    ax_info = [('%s%d' % (k,a),) + vt
        for a, axis_data in enumerate(axes, 1)
        for k, vt in axis_data
        if vt is not None]
    header.update(ax_info)

    # Now setup the GFREQS
    # Jitter them randomly, except for the endpoints
    frequency_jitter = np.random.random(size=gfrequency.shape)-0.5
    frequency_jitter *= 0.1*bandwidth_delta
    frequency_jitter[0] = frequency_jitter[-1] = 0.0
    gfrequency += frequency_jitter

    # Check that gfrequency is monotically increasing
    assert np.all(np.diff(gfrequency) >= 0.0)

    for i, gfreq in enumerate(gfrequency, 1):
        header['GFREQ%d' % i] = gfreq

    # Figure out the beam filenames from the schema
    filenames = _create_filenames(schema, polarisation_type)

    for filename in [f for ri_pair in list(filenames.values()) for f in ri_pair]:
        if overwrite or not pexists(filename):
            ex = np.deg2rad(1.0)
            coords = np.linspace(-ex, ex, header['NAXIS2'], endpoint=True)

            r = np.sqrt(coords[None,:,None]**2 + coords[None,None,:]**2)
            fq = gfrequency[:,None,None]
            beam = np.cos(np.minimum(65*fq*1e-9*r, 1.0881))**3

            primary_hdu = fits.PrimaryHDU(beam, header=header)
            primary_hdu.writeto(filename, overwrite=overwrite)
        else:
            with fits.open(filename, mode='update', memmap=False) as file:
                uheader = file[0].header
                assert uheader['CTYPE3'] == 'FREQ'
                unchan = uheader['NAXIS3']

                bandwidth_delta = bandwidth / (unchan-1)
                uheader['CRVAL3'] = frequency[0]
                uheader['CDELT3'] = bandwidth_delta

                # Generate a linear space of grid frequencies
                # Jitter them randomly, except for the endpoints
                gfrequency = np.linspace(frequency[0], frequency[-1],
                                        unchan-1, dtype=np.float64)
                frequency_jitter = np.random.random(size=gfrequency.shape)-0.5
                frequency_jitter *= 0.1*bandwidth_delta
                frequency_jitter[0] = frequency_jitter[-1] = 0.0
                gfrequency += frequency_jitter

                # Check that gfrequency is monotically increasing
                assert np.all(np.diff(gfrequency) >= 0.0)

                # Remove existing GFREQ data
                try:
                    del uheader['GFREQ?*']
                except:
                    pass

                # Update existing GFREQ data
                for i, gfreq in enumerate(gfrequency, 1):
                    uheader['GFREQ%d' % i] = gfreq

    return filenames

if __name__ == "__main__":
    filenames = beam_factory(schema="test_beam_$(corr)_$(reim).fits")
