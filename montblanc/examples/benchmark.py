import argparse
import logging
import time

import dask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def create_parser():
    """ Create script argument parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", type=str, default="threaded",
                                    help="'threaded', 'multiprocessing', "
                                        "or distributed scheduler adddress "
                                        " 'tcp://202.192.33.166:8786'")
    parser.add_argument("-b", "--budget", type=int, required=False, default=2*1024**3,
                                    help="Memory budget for solving a portion of the RIME")
    parser.add_argument("-nt", "--timesteps", type=int, required=False, default=1000,
                                    help="Number of timesteps")
    parser.add_argument("-na", "--antenna", type=int, required=False, default=64,
                                    help="Number of antenna")
    parser.add_argument("-nc", "--channels", type=int, required=False, default=64,
                                    help="Number of channels")
    parser.add_argument("-np", "--point", type=int, required=False, default=100,
                                    help="Number of point sources")
    parser.add_argument("-ng", "--gaussian", type=int, required=False, default=0,
                                    help="Number of gaussian sources")

    parser.add_argument("-i", "--iterations", type=int, required=False, default=10,
                                    help="Number of timing iterations")
    return parser

args = create_parser().parse_args()

def set_scheduler(args):
    """ Set the scheduler to use, based on the script arguments """
    import dask
    if args.scheduler in ("mt", "thread", "threaded", "threading"):
        logging.info("Using multithreaded scheduler")
        dask.set_options(get=dask.threaded.get)
    elif args.scheduler in ("mp", "multiprocessing"):
        import dask.multiprocessing
        logging.info("Using multiprocessing scheduler")
        dask.set_options(get=dask.multiprocessing.get)
    else:
        import distributed

        logging.info("Using distributed scheduler with address '{}'".format(args.scheduler))
        client = distributed.Client(args.scheduler)
        client.restart()
        dask.set_options(get=client.get)

set_scheduler(args)

from montblanc.impl.rime.tensorflow.dataset import (default_dataset,
                                                rechunk_to_budget,
                                                default_chunks,
                                                _update_default_chunks)
from montblanc.impl.rime.tensorflow.dask_rime import Rime

# Set up problem default dimensions
dims = {
    'utime': args.timesteps,
    'antenna': args.antenna,
    'chan': args.channels,
    'row': args.timesteps*args.antenna*(args.antenna-1)//2,
    'point': args.point,
    'gaussian': args.gaussian,
}

# Override default dimension chunking strategy
_update_default_chunks({
    'row': 1000,
    'point': 50,
    'gaussian': 50,
    'sersic': 50,
})

# Chunk so that multiple threads/processes/workers are employed
mds = default_dataset(dims=dims)
#mds = rechunk_to_budget(mds, args.budget)

logging.info("Input data size %.3fGB" % (mds.nbytes / (1024.**3)))
logging.info(mds)

rime = Rime()
rime.set_options({'polarisation_type': 'linear', 'device_type':'CPU'})

model_vis, chi_squared = rime(mds)

iterations = 10
total_time = 0.0

for i in range(args.iterations):
    start = time.time()
    logging.info("Iteration '%d' started at '%.3f'" % (i, start))

    X2 = chi_squared.compute()

    end = time.time()
    logging.info("Iteration '%d' completed at '%.3f'" % (i, end))

    elapsed = end - start
    logging.info("Iteration '%d' computed chi-squared '%.3f' in '%.3f' seconds" % (i, X2, elapsed))

    total_time += elapsed

logging.info("Average time '%.3f'" % (total_time / args.iterations))