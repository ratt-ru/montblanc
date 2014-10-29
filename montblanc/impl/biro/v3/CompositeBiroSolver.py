import copy
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray
import pycuda.tools
import types

import montblanc

from montblanc.BaseSolver import BaseSolver
from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

from montblanc.impl.biro.v3.BiroSolver import BiroSolver

def ary_dict(name,shape,dtype,cpu=True,gpu=False):
    return {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'registrant' : 'BiroSolver',
        'gpu' : gpu,
        'cpu' : cpu,
        'shape_member' : True,
        'dtype_member' : True
    }

def prop_dict(name,dtype,default):
    return {
        'name' : name,
        'dtype' : dtype,
        'default' : default,
        'registrant' : 'BiroSolver',
        'setter' : True
    }

# Set up gaussian scaling parameters
# Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
# and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
fwhm2int = 1.0/np.sqrt(np.log(256))

# Dictionary of properties
P = {
    # Note that we don't divide by speed of light here. meqtrees code operates
    # on frequency, while we're dealing with wavelengths.
    'gauss_scale' : prop_dict('gauss_scale', 'ft', fwhm2int*np.sqrt(2)*np.pi),
    'ref_wave' : prop_dict('ref_wave', 'ft', 0.0),
    'sigma_sqrd' : prop_dict('sigma_sqrd', 'ft', 1.0),
    'X2' : prop_dict('X2', 'ft', 0.0),
    'beam_width' : prop_dict('beam_width', 'ft', 65),
    'beam_clip' : prop_dict('beam_clip', 'ft', 1.0881),
}

# Dictionary of arrays
A = {
    # Input Arrays
    'uvw' : ary_dict('uvw', (3,'ntime','na'), 'ft'),
    'ant_pairs' : ary_dict('ant_pairs', (2,'ntime','nbl'), np.int32),

    'lm' : ary_dict('lm', (2,'nsrc'), 'ft'),
    'brightness' : ary_dict('brightness', (5,'ntime','nsrc'), 'ft'),
    'gauss_shape' : ary_dict('gauss_shape', (3, 'ngsrc'), 'ft'),

    'wavelength' : ary_dict('wavelength', ('nchan',), 'ft'),
    'point_errors' : ary_dict('point_errors', (2,'ntime','na'), 'ft'),
    'weight_vector' : ary_dict('weight_vector', (4,'ntime','nbl','nchan'), 'ft'),
    'bayes_data' : ary_dict('bayes_data', (4,'ntime','nbl','nchan'), 'ct'),

    # Result arrays
    'jones_scalar' : ary_dict('jones_scalar', ('ntime','na','nsrc','nchan'), 'ct', cpu=False),
    'vis' : ary_dict('vis', (4,'ntime','nbl','nchan'), 'ct', cpu=False),
    'chi_sqrd_result' : ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', cpu=False),

    'X2' : ary_dict('X2', (1, ), 'ft'),
}

ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

class CompositeBiroSolver(BaseSolver):
    """
    Composite solver implementation for BIRO.

    Implements a solver composed of multiple BiroSolvers. The sub-solver
    memory transfers and pipelines are executed asynchronously.
    """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, dtype=DEFAULT_DTYPE,
        pipeline=None, **kwargs):
        """
        CompositeBiroSolver Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
            pipeline : list of nodes
                nodes defining the GPU kernels used to solve this RIME
        Keyword Arguments:
            context : pycuda.driver.Context
                CUDA context to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSolver object.
            weight_vector: boolean
                if True, use a weight vector when evaluating the
                RIME, else use a single sigma squared value.
            mem_budget: integer
                Amount of memory in bytes that the solver should
                take into account when fitting the problem onto the
                GPU.
            nsolvers: integer
                Number of sub-solvers to use when subdividing the
                problem.
        """

        super(CompositeBiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, dtype=dtype, pipeline=pipeline, **kwargs)

        A_main = copy.deepcopy(A)
        P_main = copy.deepcopy(P)

        for name, ary in A_main.iteritems():
            # Add a transfer method
            ary['transfer_method'] = self.get_transfer_method(name)

        for name, prop in P_main.iteritems():
            prop['setter_method'] = self.get_setter_method(name)

        self.register_arrays(A_main)
        self.register_properties(P_main)

        #print 'Composite Solver Memory CPU %s GPU %s ntime %s' \
        #    % (montblanc.util.fmt_bytes(self.cpu_bytes_required()),
        #    montblanc.util.fmt_bytes(self.gpu_bytes_required()),
        #    ntime)

        # Allocate CUDA constructs using the supplied context
        with self.context as ctx:
            (free_mem,total_mem) = cuda.mem_get_info()

            #(free_mem,total_mem) = cuda.mem_get_info()
            #print 'free %s total %s ntime %s' \
            #    % (montblanc.util.fmt_bytes(free_mem),
            #        montblanc.util.fmt_bytes(total_mem),
            #        ntime)

            # Work with a supplied memory budget, otherwise use
            # free memory less a small amount
            mem_budget = kwargs.get('mem_budget', free_mem-10*ONE_MB)

            # Work out how many timesteps we can fit in our memory budget
            self.vtime = self.viable_timesteps(mem_budget)

            (free_mem,total_mem) = cuda.mem_get_info()
            #print 'free %s total %s ntime %s vtime %s' \
            #    % (montblanc.util.fmt_bytes(free_mem),
            #        montblanc.util.fmt_bytes(total_mem),
            #        ntime, self.vtime)

            # They may fit in completely            
            if ntime < self.vtime: self.vtime = ntime

            # Configure the number of solvers used
            self.nsolvers = kwargs.get('nsolvers', 2)
            self.time_begin = np.arange(self.nsolvers)*self.vtime//self.nsolvers
            self.time_end = np.arange(1,self.nsolvers+1)*self.vtime//self.nsolvers
            self.time_diff = self.time_end - self.time_begin

            #print 'time begin %s' % self.time_begin
            #print 'time end %s' % self.time_end
            #print 'time diff %s' % self.time_end

            # Create streams and events for the solvers
            self.stream = [cuda.Stream() for i in range(self.nsolvers)]
            self.transfer_start = [cuda.Event(cuda.event_flags.DISABLE_TIMING)
                for i in range(self.nsolvers)]
            self.transfer_end = [cuda.Event(cuda.event_flags.DISABLE_TIMING)
                for i in range(self.nsolvers)]

            # Create a memory pool for PyCUDA gpuarray.sum reduction results
            self.dev_mem_pool = pycuda.tools.DeviceMemoryPool()
            self.dev_mem_pool.allocate(10*ONE_KB).free()

            # Create a pinned memory pool for asynchronous transfer to GPU arrays
            self.pinned_mem_pool = pycuda.tools.PageLockedMemoryPool()
            self.pinned_mem_pool.allocate(shape=(10*ONE_KB,),dtype=np.int8).base.free()

        # Create the sub-solvers
        self.solvers = [BiroSolver(na=na,
            nchan=nchan, ntime=self.time_diff[i],
            npsrc=npsrc, ngsrc=ngsrc,
            dtype=dtype, pipeline=copy.deepcopy(pipeline),
            **kwargs) for i in range(self.nsolvers)]

        # Register arrays on the sub-solvers
        A_sub = copy.deepcopy(A)
        P_sub = copy.deepcopy(P)

        for name, ary in A_sub.iteritems():
            # Add a transfer method
            ary['transfer_method'] = self.get_sub_transfer_method(name)
            ary['cpu'] = False
            ary['gpu'] = True

        for i, slvr in enumerate(self.solvers):
            slvr.register_arrays(A_sub)
            slvr.register_properties(P_sub)
            # Indicate that all numpy arrays on the CompositeSolver
            # have been transferred to the sub-solvers
            slvr.was_transferred = {}.fromkeys(
                [v.name for v in self.arrays.itervalues()], True)

            #print 'Sub-solver %s Memory CPU %s GPU %s ntime %s' \
            #    % (i,
            #    montblanc.util.fmt_bytes(slvr.cpu_bytes_required()),
            #    montblanc.util.fmt_bytes(slvr.gpu_bytes_required()),
            #    slvr.ntime)

        #(free_mem,total_mem) = cuda.mem_get_info()
        #print 'free %s total %s ntime %s vtime %s' \
        #    % (montblanc.util.fmt_bytes(free_mem),
        #        montblanc.util.fmt_bytes(total_mem),
        #        ntime, self.vtime)

        self.use_weight_vector = kwargs.get('weight_vector', False)

    def transfer_arrays(self, sub_solver_idx, time_begin, time_end):
        """
        Transfer CPU arrays on the CompositeBiroSolver over to the
        BIRO sub-solvers asynchronously. A pinned memory pool is used
        to store the data temporarily.

        Arrays without a time dimension are transferred as a whole,
        otherwise the time section of the array appropriate to the
        sub-solver is transferred.
        """
        i = sub_solver_idx
        subslvr = self.solvers[i]
        stream = self.stream[i]

        self.transfer_start[i].record(stream=stream)

        for r in self.arrays.itervalues():
            # The array has been transferred, try the next one
            if subslvr.was_transferred[r.name]: continue

            cpu_name = montblanc.util.cpu_name(r.name)
            gpu_name = montblanc.util.gpu_name(r.name)

            # Get the CPU array on the composite solver
            # and the CPU array and the GPU array
            # on the sub-solver
            cpu_ary = getattr(self,cpu_name)
            gpu_ary = getattr(subslvr,gpu_name)

            # Set up the slicing of the main CPU array. It we're dealing with the
            # time dimension, slice the appropriate chunk, otherwise take
            # everything
            all_slice = slice(None,None,1)
            time_slice = slice(time_begin, time_end ,1)
            cpu_idx = tuple([time_slice if s == 'ntime' else all_slice
                for s in r.sshape])

            # Similarly, set up the slicing of the pinned CPU array
            time_diff = time_end - time_begin
            time_diff_slice = slice(None,time_diff,1)
            pin_idx = tuple([time_diff_slice if s == 'ntime' else all_slice
                for s in r.sshape])

            # Get a pinned array for asynchronous transfers
            cpu_shape_name = montblanc.util.shape_name(r.name)
            cpu_dtype_name = montblanc.util.dtype_name(r.name)

            pinned_ary = self.pinned_mem_pool.allocate(
                shape=getattr(subslvr,cpu_shape_name),
                dtype=getattr(subslvr,cpu_dtype_name))

            # Copy the numpy array into the pinned array
            # and perform the asynchronous transfer
            pinned_ary[pin_idx] = cpu_ary[cpu_idx]

            with self.context as ctx:
                gpu_ary.set_async(pinned_ary,stream=stream)

            # If we're not dealing with an array with a time dimension
            # then the transfer happens in one go. Otherwise, we're transferring
            # chunks of the the array and we're only done when
            # we transfer the last chunk
            if r.sshape.count('ntime') == 0 or time_end == self.ntime:
                subslvr.was_transferred[r.name] = True

        self.transfer_end[i].record(stream=stream)

    def __enter__(self):
        """
        When entering a run-time context related to this solver,
        initialise and return it.
        """
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        """
        When exiting a run-time context related to this solver,
        also perform exit for the sub-solvers.
        """
        self.shutdown()

    def initialise(self):
        """ Initialise the sub-solver """
        with self.context as ctx:
            for i, slvr in enumerate(self.solvers):
                if not slvr.pipeline.is_initialised():
                    slvr.pipeline.initialise(slvr, self.stream[i])

    def solve(self):
        """ Solve the RIME """
        self.X2 = 0.0

        with self.context as ctx:
            # Initialise the pipelines if necessary
            for i, subslvr in enumerate(self.solvers):
                if not subslvr.pipeline.is_initialised():
                    montblanc.log.warn(('Sub-solver %d not initialised'
                        ' in CompositeBiroSolver. Initialising!') % i)
                    subslvr.pipeline.initialise(subslvr, self.stream[i])

            t = 0
            result = []

            while t < self.ntime:
                t_begin = t

                # Perform any memory transfers needed
                # and execute the pipeline
                for i, subslvr in enumerate(self.solvers):
                    # Clip the timestep ending point at the total number of timesteps
                    # otherwise add the sub-solvers timesteps to t_begin
                    t_end = self.ntime if t_begin+subslvr.ntime > self.ntime \
                        else t_begin+subslvr.ntime
                    self.transfer_arrays(i, t_begin, t_end)
                    t_begin = t_end
                    subslvr.pipeline.execute(subslvr, stream=self.stream[i])

                t_begin = t

                # Get the chi squared result for each solver.
                for i, subslvr in enumerate(self.solvers):
                    # Clip the timestep ending point at the total number of timesteps
                    # otherwise add the sub-solvers timesteps to t_begin
                    t_end = self.ntime if t_begin+subslvr.ntime > self.ntime \
                        else t_begin+subslvr.ntime
                    t_diff = t_end - t_begin

                    # Allocate a temporary pinned array on the sub-solver
                    # Free and delete later
                    subslvr.X2_tmp = self.pinned_mem_pool.allocate(
                        shape=self.X2_shape, dtype=self.X2_dtype)

                    # Swap the allocators for the reduction, since it uses
                    # chi_sqrd_result_gpu's allocator internally
                    tmp_alloc = subslvr.chi_sqrd_result_gpu.allocator
                    subslvr.chi_sqrd_result_gpu.allocator = self.dev_mem_pool.allocate

                    # OK, perform the reduction over the appropriate timestep section
                    # of the chi squared terms.
                    pycuda.gpuarray.sum(
                        subslvr.chi_sqrd_result_gpu[:t_diff,:],
                        stream=self.stream[i]) \
                            .get_async(
                                ary=subslvr.X2_tmp,
                                stream=self.stream[i])

                    # Advance
                    subslvr.chi_sqrd_result_gpu.allocator = tmp_alloc

                    t_begin = t_end

                # Get the chi squared result for each solver.
                for i, subslvr in enumerate(self.solvers):
                    # Wait for this stream to finish executing
                    self.stream[i].synchronize()

                    # Divide by sigma squared if necessary
                    if not self.use_weight_vector:
                        self.X2 += subslvr.X2_tmp[0] / subslvr.sigma_sqrd
                    else:
                        self.X2 += subslvr.X2_tmp[0]

                    # force release of pinned memory allocated for X2_tmp
                    # and remove it from the sub-solver
                    subslvr.X2_tmp.base.free()
                    del subslvr.X2_tmp

                t = t_end

    def shutdown(self):
        """ Shutdown the solver """
        with self.context as ctx:
            for i, slvr in enumerate(self.solvers):
                slvr.pipeline.shutdown(slvr, self.stream[i])

    def get_setter_method(self,name):
        """
        Setter method for CompositeBiroSolver properties. Sets the property
        on sub-solvers.
        """

        def setter(self, value):
            setattr(self,name,value)
            for slvr in self.solvers:
                setter_method_name = montblanc.util.setter_name(name)
                setter_method = getattr(slvr,setter_method_name)
                setter_method(value)

        return types.MethodType(setter,self)

    def get_sub_transfer_method(self,name):
        def f(self, npary):
            raise Exception, 'Its illegal to call set methods on the sub-solvers'
        return types.MethodType(f,self)

    def get_transfer_method(self, name):
        """
        Transfer method for CompositeBiroSolver arrays. Sets the cpu array
        on the CompositeBiroSolver and indicates that it hasn't been transferred
        to the sub-solver
        """
        cpu_name = montblanc.util.cpu_name(name)
        def transfer(self, npary):
            self.check_array(name, npary)
            setattr(self, cpu_name, npary)
            # Indicate that this data has not been transferred to
            # the sub-solvers
            for slvr in self.solvers: slvr.was_transferred[name] = False

        return types.MethodType(transfer,self)

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, ntime, nbl), dtype=np.int32]) containing the
        default antenna pairs for each timestep at each baseline.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna. 
        slvr = self

        return np.tile(np.int32(np.triu_indices(slvr.na,1)),
            slvr.ntime).reshape(2,slvr.ntime,slvr.nbl)

    def get_flat_ap_idx(self, src=False, chan=False):
        """
        Returns a flattened antenna pair index

        Parameters
        ----------
        src : boolean
            Expand the index over the source dimension
        chan : boolean
            Expand the index over the channel dimension
        """
        # TODO: Test for src=False and chan=True, and src=True and chan=False
        # This works for
        # - src=True and chan=True.
        # - src=False and chan=False.

        # The flattened antenna pair array will look something like this.
        # It is based on 2 x ntime x nbl. Here we have 3 baselines and
        # 4 timesteps.
        #
        #            timestep
        #       0 0 0 1 1 1 2 2 2 3 3 3
        #
        # ant1: 0 0 1 0 0 1 0 0 1 0 0 1
        # ant2: 1 2 2 1 2 2 1 2 2 1 2 2

        slvr = self        
        ap = slvr.get_default_ant_pairs().reshape(2,slvr.ntime*slvr.nbl)

        C = 1

        if src is True: C *= slvr.nsrc
        if chan is True: C *= slvr.nchan

        repeat = np.repeat(np.arange(slvr.ntime),slvr.nbl)*slvr.na*C

        ant0 = ap[0]*C + repeat
        ant1 = ap[1]*C + repeat

        if src is True or chan is True:
            tile = np.tile(np.arange(C),slvr.ntime*slvr.nbl) 

            ant0 = np.repeat(ant0, C) + tile
            ant1 = np.repeat(ant1, C) + tile

        return ant0, ant1