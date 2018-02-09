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
                    dtype=np.float64,
                    schema=DEFAULT_SCHEMA,
                    overwrite=True):
    """ Generate a MeqTrees compliant beam cube """

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
        ("NAXIS", (513, "number of X")),
        ("CRPIX", (257, "reference pixel (one relative)")),
        ("CRVAL", (0.0110828777007, "degrees")),
        ("CDELT", (0.011082, "degrees"))]

    axis2 = [
        ("CTYPE", ('Y', "points up on the sky")),
        ("CUNIT", ('DEG', 'degrees')),
        ("NAXIS", (513, "number of Y")),
        ("CRPIX", (257, "reference pixel (one relative)")),
        ("CRVAL", (-2.14349358381E-07, "degrees")),
        ("CDELT", (0.011082, "degrees"))]

    axis3 = [
        ("CTYPE", ('FREQ', )),
        ("CUNIT", None),
        ("NAXIS", (33, "number of FREQ")),
        ("CRPIX", (1, "reference frequency position")),
        ("CRVAL", (1400062500.0, "reference frequency")),
        ("CDELT", (246093.75, "frequency step in Hz"))]

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

    # Figure out the beam filenames from the schema
    filenames = _create_filenames(schema, polarisation_type)

    shape = tuple(reversed([nax[1][0] for _, _, nax, _, _, _ in axes
                                            if nax[1] is not None]))

    for filename in [f for ri_pair in filenames.values() for f in ri_pair]:
        beam = np.random.random(size=shape).astype(dtype)

        primary_hdu = fits.PrimaryHDU(beam, header=header)
        primary_hdu.writeto(filename, overwrite=overwrite)

    return filenames

if __name__ == "__main__":
    filenames = beam_factory(schema="test_beam_$(corr)_$(reim).fits")
