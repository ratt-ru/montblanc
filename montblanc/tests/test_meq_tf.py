"""
This script runs Montblanc and MeqTrees and compares
the visibilities output by both.

To run this script you'll need a Measurement Set.
It's often easier to create this using the
`simms <https://github.com/radio-astro/simms_>`_ package.

For example, create a VLA Measurement Set, call the following:

.. code-block:: shell

    simms -T vla -t ascii -cs itrf -st 1 -dt 4 -f0 1.42GHz -df 4MHz
        -nc 8 -dir "J2000,5h42m36.1378s,+49d51m7.23000000001s"
        -feed 'perfect R L' -pl "RR RL LR LL" -n vla_test.ms
        ~/paper_sims/simms/simms/observatories/vlac.itrf.txt

Then, to call the script run

.. code-block:: shell

    python test_meq_tf.py vla.ms -p circular -a "overwrite_beams=True"

This tells the script to configure Montblanc+MeqTrees for circular
polarisations (the VLA telescope uses circular polarisations) and
to create new cos**3 testing beams. It is possible to also supply
your own beams for the test by specifying
``-a "beam_file_schema='my_beam_\$(corr)_\$(reim).fits'`` for example.

Other options can be passed

-- See :func:`run_test` for more details
and options.
"""


import itertools
import os
import subprocess
import sys
import tempfile

from astropy.io import fits
import numpy as np
import pyrap.tables as pt

rf = np.random.random

from montblanc.tests.beam_factory import beam_factory

# Directory that holds MS and Beam data
DATA_DIR = 'data'

def run_test(msfile="/mb_testing/data/mk64.Lwide.0.5hr.30s.856mhz.ms", 
             pol_type="linear", **kwargs):
    """
    Parameters
    ----------
    msfile : str
        Name of the Measurement Set
    pol_type : str
        'linear' or 'circular'
    beam_file_schema (optional) : str
        Beam filename schema. Defaults to 'test_beam_$(corr)_$(reim).fits'
    overwrite_beams (optional) : bool
        If ``True`` create new beams using the cos**3 beam
    """


    #=========================================
    # Directory and Script Configuration
    #=========================================

    # Directory in which we expect our measurement set to be located
    meq_vis_column = 'MODEL_DATA'
    mb_vis_column = 'CORRECTED_DATA'

    # Directory in which meqtree-related files are read/written
    import os
    meq_dir = os.path.join(os.path.dirname(__file__), 'meqtrees')
    # Scripts
    meqpipe = 'meqtree-pipeliner.py'
    # Meqtree profile and script
    cfg_file = os.path.join(meq_dir, 'tdlconf.profiles')
    sim_script = os.path.join(meq_dir, 'turbo-sim.py')
    tigger_sky_file = os.path.join(meq_dir, 'sky_model.txt')

    # Is the beam enabled
    beam_on = kwargs.get('beam_on', True)
    beam_on = 1 if beam_on is True else 0

    # Directory in which we expect our beams to be located
    beam_file_schema = 'test_beam_$(corr)_$(reim).fits'

    # Beam file pattern
    beam_file_schema = kwargs.get("beam_file_schema", beam_file_schema)

    l_axis = kwargs.get('l_axis', '-X')
    m_axis = kwargs.get('m_axis', 'Y')

    # Find the location of the meqtree pipeliner script
    meqpipe_actual = subprocess.check_output(['which', meqpipe]).strip()
    cfg_section = '-'.join(('montblanc', 'compare', pol_type))

    #======================================================
    # Configure the beam files with frequencies from the MS
    #======================================================

    from montblanc.impl.rime.tensorflow.sources.fits_beam_source_provider import (
        _create_filenames, _open_fits_files)

    # Zero the visibility data
    with pt.table(msfile, ack=False, readonly=False) as T:
        data_desc = T.getcoldesc('DATA')

        try:
            shape = data_desc['shape'].tolist()
        except KeyError:
            shape = list(T.getcol('DATA', startrow=0, nrow=1).shape[1:])

        shape = [T.nrows()] + shape
        T.putcol(mb_vis_column, np.zeros(shape, dtype=np.complex64))
        T.putcol(meq_vis_column, np.zeros(shape, dtype=np.complex64))

    # Extract frequencies from the MS
    with pt.table(msfile + '::SPECTRAL_WINDOW', ack=False) as SW:
        frequency = SW.getcol('CHAN_FREQ')[0]

    bandwidth = frequency[-1] - frequency[0]

    overwrite_beams = kwargs.get('overwrite_beams', False)

    # Get filenames from pattern and open the files
    filenames = beam_factory(polarisation_type=pol_type,
                            frequency=frequency,
                            schema=beam_file_schema,
                            overwrite=overwrite_beams)

    #=========================================
    # Source Configuration
    #=========================================

    np.random.seed(0)
    dtype = np.float64
    ctype = np.complex128 if dtype == np.float64 else np.complex64

    def get_point_sources(nsrc):
        source_coords = np.empty(shape=(nsrc, 2), dtype=dtype)
        stokes = np.empty(shape=(nsrc, 4), dtype=dtype)
        I, Q, U, V = stokes[:,0], stokes[:,1], stokes[:,2], stokes[:,3]
        alphas = np.empty(shape=(nsrc,), dtype=dtype)
        ref_freq = np.empty(shape=(nsrc,), dtype=dtype)

        # Source coordinates between -45 and 45 degrees
        source_coords[:] = (rf(size=source_coords.shape) - 0.5)*90.0
        Q[:] = rf(size=Q.shape)*0.1
        U[:] = rf(size=U.shape)*0.1
        V[:] = rf(size=V.shape)*0.1
        I[:] = np.sqrt(Q**2 + U**2 + V**2)*1.5 + rf(size=I.shape)*0.1

        # Zero and invert selected stokes parameters
        if nsrc > 0:
            zero_srcs = np.random.randint(nsrc, size=(2,))
            source_coords[zero_srcs,:] = 0

            # Create sources with both positive and negative flux
            sign = 2*np.random.randint(2, size=I.shape) - 1
            I[:] *= sign

        alphas[:] = 2*(np.random.random(size=alphas.size) - 0.5)

        ref_freq[:] = 1.3e9 + np.random.random(ref_freq.size)*0.2e9

        return (np.deg2rad(source_coords), np.asarray(stokes),
                np.asarray(alphas), np.asarray(ref_freq))

    def get_gaussian_sources(nsrc):
        c, s, a, r= get_point_sources(nsrc)
        gauss_shape = np.empty(shape=(3, nsrc), dtype=np.float64)
        gauss_shape[:] = rf(size=gauss_shape.shape)
        return c, s, a, r, gauss_shape

    npsrc, ngsrc = 10, 10

    pt_lm, pt_stokes, pt_alpha, pt_ref_freq = get_point_sources(npsrc)

    assert pt_lm.shape == (npsrc, 2), pt_lm.shape
    assert pt_stokes.shape == (npsrc, 4), pt_stokes.shape
    assert pt_alpha.shape == (npsrc,), pt_alpha.shape
    assert pt_ref_freq.shape == (npsrc,), pt_ref_freq.shape

    g_lm, g_stokes, g_alpha, g_ref_freq, g_shape = get_gaussian_sources(ngsrc)

    #=========================================
    # Create Tigger ASCII sky model
    #=========================================

    from Tigger.Models.Formats.AIPSCCFITS import lm_to_radec

    # Need the phase centre for lm_to_radec
    with pt.table(msfile + '::FIELD', ack=False, readonly=True) as F:
        ra0, dec0 = F.getcol('PHASE_DIR')[0][0]

    # Create the tigger sky model
    with open(tigger_sky_file, 'w') as f:
        f.write('#format: ra_d dec_d i q u v spi freq0 emaj_s emin_s pa_d\n')
        it = enumerate(zip(pt_lm, pt_stokes, pt_alpha, pt_ref_freq))
        for i, ((l, m), (I, Q, U, V), alpha, ref_freq) in it:
            ra, dec = lm_to_radec(l, m, ra0, dec0)
            l, m = np.rad2deg([ra,dec])

            f.write('{l:.20f} {m:.20f} {i} {q} {u} {v} {spi} {rf:.20f}\n'.format(
                l=l, m=m, i=I, q=Q, u=U, v=V, spi=alpha, rf=ref_freq))

        it = enumerate(zip(g_lm, g_stokes, g_alpha, g_ref_freq, g_shape.T))
        for i, ((l, m), (I, Q, U, V), alpha, ref_freq, (emaj, emin, pa)) in it:
            ra, dec = lm_to_radec(l, m, ra0, dec0)
            l, m = np.rad2deg([ra,dec])
            # Convert to seconds
            emaj, emin = np.asarray([emaj, emin])*648000./np.pi
            # Convert to degrees
            pa *= 180.0/np.pi

            f.write('{l:.20f} {m:.20f} {i} {q} {u} {v} {spi} {rf:.20f} '
                    '{emaj} {emin} {pa}\n'.format(
                        l=l, m=m, i=I, q=Q, u=U, v=V, spi=alpha, rf=ref_freq,
                        emaj=emaj, emin=emin, pa=pa))

#=========================================
# Call MeqTrees
#=========================================

    #=========================================
    # Call MeqTrees
    #=========================================

    cmd_list = ['python',
        # Meqtree Pipeline script
        meqpipe_actual,
        # Configuration File
        '-c', cfg_file,
        # Configuration section
        '[{section}]'.format(section=cfg_section),
        # Enable the beam?
        'me.e_enable = {e}'.format(e=beam_on),
        # Measurement Set
        'ms_sel.msname={ms}'.format(ms=msfile),
        # Tigger sky file
        'tiggerlsm.filename={sm}'.format(sm=tigger_sky_file),
        # Output column
        'ms_sel.output_column={c}'.format(c=meq_vis_column),
        # Imaging Column
        'img_sel.imaging_column={c}'.format(c=meq_vis_column),
        # Beam FITS file pattern
        'pybeams_fits.filename_pattern={p}'.format(p=beam_file_schema),
        # FITS L and M AXIS
        'pybeams_fits.l_axis={l}'.format(l=l_axis),
        'pybeams_fits.m_axis={m}'.format(m=m_axis),
        sim_script,
        '=simulate'
        ]

    import montblanc

    from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
    from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
        MSSourceProvider,
        FitsBeamSourceProvider,
        CachedSourceProvider)

    from montblanc.impl.rime.tensorflow.sinks import MSSinkProvider

    class RadioSourceProvider(SourceProvider):
        def name(self):
            return "RadioSourceProvider"

        def point_lm(self, context):
            lp, up = context.dim_extents('npsrc')
            return pt_lm[lp:up, :]

        def point_stokes(self, context):
            (lp, up), (lt, ut), (lc, uc) = context.dim_extents('npsrc', 'ntime', 'nchan')
            # (npsrc, ntime, nchan, 4)
            s = pt_stokes[lp:up,None,None,:]
            a = np.broadcast_to(pt_alpha[lp:up,None,None,None], (up-lp,ut-lt,1,1))
            rf = pt_ref_freq[lp:up,None,None,None]
            f = frequency[None,None,lc:uc,None]

            return s*(f/rf)**a

        def gaussian_lm(self, context):
            lg, ug = context.dim_extents('ngsrc')
            return g_lm[lg:ug, :]

        def gaussian_stokes(self, context):
            (lg, ug), (lt, ut), (lc, uc) = context.dim_extents('ngsrc', 'ntime', 'nchan')
            # (ngsrc, ntime, nchan, 4)
            s = g_stokes[lg:ug,None,None,:]
            a = np.broadcast_to(pt_alpha[lg:ug,None,None,None], (ug-lg,ut-lt,1,1))
            rf = g_ref_freq[lg:ug,None,None,None]
            f = frequency[None,None,lc:uc,None]

            return s*(f/rf)**a

        def gaussian_shape(self, context):
            (lg, ug) = context.dim_extents('ngsrc')
            gauss_shape = g_shape[:,lg:ug]
            emaj = gauss_shape[0]
            emin = gauss_shape[1]
            pa = gauss_shape[2]

            gauss = np.empty(context.shape, dtype=context.dtype)

            gauss[0,:] = emaj * np.sin(pa)
            gauss[1,:] = emaj * np.cos(pa)
            emaj[emaj == 0.0] = 1.0
            gauss[2,:] = emin / emaj

            return gauss

        def updated_dimensions(self):
            return [('npsrc', pt_lm.shape[0]), ('ngsrc', g_lm.shape[0])]

    slvr_cfg = montblanc.rime_solver_cfg(
        mem_budget=1024*1024*1024,
        data_source='default',
        dtype='double' if dtype == np.float64 else 'float',
        polarisation_type=pol_type,
        auto_correlations=False,
        version='tf')

    slvr = montblanc.rime_solver(slvr_cfg)

    ms_mgr = MeasurementSetManager(msfile, slvr_cfg)

    source_providers = []
    source_providers.append(MSSourceProvider(ms_mgr))

    if beam_on == 1:
        beam_prov = FitsBeamSourceProvider(beam_file_schema,
            l_axis=l_axis, m_axis=m_axis)
        source_providers.append(beam_prov)

    source_providers.append(RadioSourceProvider())
    cache_prov = CachedSourceProvider(source_providers)
    source_providers = [cache_prov]

    sink_providers = [MSSinkProvider(ms_mgr, mb_vis_column)]
    slvr.solve(source_providers=source_providers,
        sink_providers=sink_providers)

    import time
    time.sleep(1)

    for obj in source_providers + sink_providers + [ms_mgr]:
        obj.close()

    # Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
    subprocess.call(cmd_list)

    # Compare MeqTree and Montblanc visibilities
    with pt.table(msfile, ack=False, readonly=True) as MS:
        ntime, nbl, nchan = slvr.hypercube.dim_global_size('ntime', 'nbl', 'nchan')
        shape = (ntime, nbl, nchan, 4)
        meq_vis = MS.getcol(meq_vis_column).reshape(shape)
        mb_vis = MS.getcol(mb_vis_column).reshape(shape)

        # Compare
        close = np.isclose(meq_vis, mb_vis)
        not_close = np.invert(close)
        problems = np.nonzero(not_close)

        # Everything agrees, exit
        if problems[0].size == 0:
            print('Montblanc and MeqTree visibilities agree')
            sys.exit(1)

        bad_vis_file = 'bad_visibilities.txt'

        # Some visibilities differ, do some analysis
        print((("Montblanc differs from MeqTrees by {nc}/{t} visibilities. "
            "Writing them out to '{bvf}'").format(
            nc=problems[0].size, t=not_close.size, bvf=bad_vis_file)))

        abs_diff = np.abs(meq_vis - mb_vis)
        rmsd = np.sqrt(np.sum(abs_diff**2)/abs_diff.size)
        nrmsd = rmsd / (np.max(abs_diff) - np.min(abs_diff))
        print(('RMSD {rmsd} NRMSD {nrmsd}'.format(rmsd=rmsd, nrmsd=nrmsd)))

        # Plot a histogram of the difference
        try:
            import matplotlib
            matplotlib.use('pdf')
            import matplotlib.pyplot as plt
        except:
            print(("Exception importing matplotlib %s" % sys.exc_info()[2]))
        else:
            try:
                nr_of_bins = 100
                n, bins, patches = plt.hist(abs_diff.flatten(),
                    bins=np.logspace(np.log10(1e-10), np.log10(1.0), nr_of_bins))

                plt.gca().set_xscale("log")
                plt.xlabel('Magnitude Difference')
                plt.ylabel('Counts')
                plt.grid(True)

                plt.savefig('histogram.pdf')
            except:
                print(("Error plotting histogram %s" % sys.exc_info()[2]))

        mb_problems = mb_vis[problems]
        meq_problems = meq_vis[problems]
        difference = mb_problems - meq_problems
        amplitude = np.abs(difference)

        # Create an iterator over the first 100 problematic visibilities
        t = (np.asarray(problems).T, mb_problems, meq_problems, difference, amplitude)
        it = enumerate(zip(*t))
        it = itertools.islice(it, 0, 1000, 1)

        # Write out the problematic visibilities to file
        with open(bad_vis_file, 'w') as f:
            for i, (p, mb, meq, d, amp) in it:
                f.write("{i} {t} Montblanc: {mb} MeqTrees: {meq} "
                    "Difference {d} Absolute Difference {ad} \n".format(
                        i=i, t=p, mb=mb, meq=meq, d=d, ad=amp))


if __name__ == "__main__":
    import argparse
    from os.path import join as pjoin

    from montblanc.util import parse_python_assigns

    def create_parser():
        p = argparse.ArgumentParser()
        p.add_argument("ms", default=pjoin("data", "mk64.Lwide.0.5hr.30s.856mhz.ms"),
                                nargs="?")
        p.add_argument("-p", "--polarisation-type",
                            choices=['linear', 'circular'],
                            default='linear')
        p.add_argument("-a", "--args", type=parse_python_assigns,
                            default="",
                            help="semi-colon separated list of "
                                "python variable assignments. "
                                "These variable assignments are "
                                "passed into the run_tests function."
                                "See its docstring for details.")
        return p

    args = create_parser().parse_args()

    run_test(args.ms, args.polarisation_type, **args.args)

