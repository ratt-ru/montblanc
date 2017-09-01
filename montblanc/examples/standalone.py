import argparse

import numpy as np

import montblanc
from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntime", default=100, type=int,
                                   help="Number of timesteps")
    parser.add_argument("--nchan", default=64, type=int,
                                   help="Number of channels")
    parser.add_argument("--na", default=27, type=int,
                                    help="Number of antenna")

    return parser

args = create_parser().parse_args()

# Two point sources at centre

# LM coordinates
lm_coords = [(0.0, 0.0),
             (0.0, 0.0)]
# Stokes parameters (I, Q, U, V)
lm_stokes = [(1.0, 0.0, 0.0, 0.0),
             (1.0, 0.0, 0.0, 0.0)]

class CustomSourceProvider(SourceProvider):
    """
    Supplies data to montblanc via data source methods,
    which have the following signature.

    .. code-block:: python

        def point_lm(self, context)
            ...
    """
    def name(self):
        """ Name of Source Provider """
        return self.__class__.__name__

    def updated_dimensions(self):
        """ Inform montblanc about dimension sizes """
        return [("ntime", args.ntime),      # Timesteps
                ("nchan", args.nchan),      # Channels
                ("na", args.na),            # Antenna
                ("npsrc", len(lm_coords))]  # Number of point sources

    def point_lm(self, context):
        """ Supply point source lm coordinates to montblanc """

        # Shape (npsrc, 2)
        (ls, us), _ = context.array_extents(context.name)
        return np.asarray(lm_coords[ls:us], dtype=context.dtype)

    def point_stokes(self, context):
        """ Supply point source stokes parameters to montblanc """

        # Shape (npsrc, ntime, 4)
        (ls, us), (lt, ut), (l, u) = context.array_extents(context.name)

        data = np.empty(context.shape, context.dtype)
        data[ls:us,:,l:u] = np.asarray(lm_stokes)[ls:us,None,:]
        return data

    def uvw(self, context):
        """ Supply UVW antenna coordinates to montblanc """

        # Shape (ntime, na, 3)
        (lt, ut), (la, ua), (l, u) = context.array_extents(context.name)

        # Create empty UVW coordinates
        data = np.empty(context.shape, context.dtype)
        data[:,:,0] = np.arange(la+1, ua+1)    # U = antenna index
        data[:,:,1] = 0                        # V = 0
        data[:,:,2] = 0                        # W = 0

        return data

class CustomSinkProvider(SinkProvider):
    """
    Receives data from montblanc via data sink methods,
    which have the following signature

    .. code-block:: python

        def model_vis(self, context):
            print context. data
    """
    def name(self):
        """ Name of the Sink Provider """
        return self.__class__.__name__

    def model_vis(self, context):
        """ Receive model visibilities from Montblanc in `context.data` """
        print context.data

# Configure montblanc solver with a memory budget of 2GB
# and set it to double precision floating point accuracy
slvr_cfg = montblanc.rime_solver_cfg(
    mem_budget=2*1024*1024*1024,
    dtype='double')

# Create montblanc solver
with montblanc.rime_solver(slvr_cfg) as slvr:
    # Create Customer Source and Sink Providers
    source_provs = [CustomSourceProvider()]
    sink_provs = [CustomSinkProvider()]

    # Call solver, supplying source and sink providers
    slvr.solve(source_providers=source_provs,
            sink_providers=sink_provs)