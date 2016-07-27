#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import itertools
import os

from attrdict import AttrDict
import concurrent.futures as cf
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import montblanc
import montblanc.util as mbu


import hypercube.util as hcu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

rime_lib_path = os.path.join(montblanc.get_montblanc_path(),
    'tensorflow', 'rime_ops', 'rime.so')
rime = tf.load_op_library(rime_lib_path)

class RimeSolver(MontblancTensorflowSolver):
    """ RIME Solver Implementation """

    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        super(RimeSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube nu depth')

        # Monkey patch these functions onto the object
        from montblanc.impl.rime.tensorflow.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)
   
        from montblanc.impl.rime.tensorflow.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        # Find out which dimensions have been modified by budgeting
        # and update them
        modded_dims = self._budget(A, slvr_cfg)

        for k, v in modded_dims.iteritems():
            self.update_dimension(k, local_size=v,
                lower_extent=0, upper_extent=v)

        # Get the data source (defaults or test data)
        data_source = slvr_cfg.get(Options.DATA_SOURCE)

        # Set up the queue data sources. Just take from
        # the defaults if the original data source was MS
        # we only want the data source types for configuring
        # the queue
        queue_data_source = (Options.DATA_SOURCE_DEFAULT
            if data_source == Options.DATA_SOURCE_MS
            else data_source)

        montblanc.log.info("Taking queue defaults from data source '{ds}'"
            .format(ds=queue_data_source))

        # Obtain default data sources for each array,
        # then update with any data sources supplied by the user
        ds = { n: (a.get(queue_data_source), a.dtype)
            for n, a in self.arrays().iteritems() }
        ds.update(slvr_cfg.get('supplied', {}))

        # Test data sources here
        ary_descs = self.arrays(reify=True)

        """
        for n, (s, t) in ds.iteritems():
            print "Testing source '{ds}' for array '{n}' with shape {s}".format(
                n=n, ds=queue_data_source, s=ary_descs[n].shape)
            if s is not None:
                a = s(self, ary_descs[n])
                print a.flatten()[0:10]
        """

        QUEUE_SIZE = 10

        from montblanc.impl.rime.tensorflow.feeders.queue_wrapper import create_queue_wrapper

        self._uvw_queue = create_queue_wrapper(QUEUE_SIZE,
            ['uvw', 'antenna1', 'antenna2'], ds)

        self._observation_queue = create_queue_wrapper(QUEUE_SIZE,
            ['observed_vis', 'flag', 'weight'], ds)

        self._frequency_queue = create_queue_wrapper(QUEUE_SIZE,
            ['frequency', 'ref_frequency'], ds)

        self._die_queue = create_queue_wrapper(QUEUE_SIZE,
            ['gterm'], ds)

        self._dde_queue = create_queue_wrapper(QUEUE_SIZE,
            ['ebeam', 'antenna_scaling', 'point_errors'], ds)

        self._point_source_queue = create_queue_wrapper(QUEUE_SIZE,
            ['point_lm', 'point_stokes', 'point_alpha'], ds)

        self._gaussian_source_queue = create_queue_wrapper(QUEUE_SIZE,
            ['gaussian_lm', 'gaussian_stokes', 'gaussian_alpha', 'gaussian_shape'], ds)

        self._sersic_source_queue = create_queue_wrapper(QUEUE_SIZE,
            ['sersic_lm', 'sersic_stokes', 'sersic_alpha', 'sersic_shape'], ds)

        self._input_queue = create_queue_wrapper(QUEUE_SIZE,
            ['model_vis'], ds)

        self._output_queue = create_queue_wrapper(QUEUE_SIZE,
            ['model_vis'], ds)

        self._data_sources = ds
        self._feed_executor = cf.ThreadPoolExecutor(1)
        self._compute_executor = cf.ThreadPoolExecutor(1)

        self._tf_session = tf.Session()

        self._src_ph_vars = AttrDict({
            n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
            for n in mbu.source_nr_vars() })

        self._property_ph_vars = AttrDict({
            n: tf.placeholder(dtype=p.dtype, shape=(), name=n)
            for n, p in self.properties().iteritems() })

        self._tf_expr = self._construct_tensorflow_expression()
            
        self._tf_session.run(tf.initialize_all_variables())

    def _budget(self, arrays, slvr_cfg):
        na = slvr_cfg.get(Options.NA)
        nsrc = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)
        src_str_list = [Options.NSRC] + mbu.source_nr_vars()
        src_reduction_str = '&'.join(['%s=%s' % (nr_var, nsrc)
            for nr_var in src_str_list])

        mem__budget = slvr_cfg.get('mem_budget', 256*ONE_MB)
        T = self.template_dict()

        # Figure out a viable dimension configuration
        # given the total problem size 
        viable, modded_dims = mbu.viable_dim_config(
            mem__budget, arrays, T, [src_reduction_str,
                'ntime',
                'nbl={na}&na={na}'.format(na=na)], 1)                

        # Create property dictionary with updated dimensions.
        # Determine memory required by our chunk size
        mT = T.copy()
        mT.update(modded_dims)
        required_mem = mbu.dict_array_bytes_required(arrays, mT)

        # Log some information about the memory _budget
        # and dimension reduction
        montblanc.log.info(("Selected a solver memory _budget of {rb} "
            "given a hard limit of {mb}.").format(
            rb=mbu.fmt_bytes(required_mem),
            mb=mbu.fmt_bytes(mem__budget)))

        montblanc.log.info((
            "The following dimension reductions "
            "have been applied:"))

        for k, v in modded_dims.iteritems():
            montblanc.log.info('{p}{d}: {id} => {rd}'.format
                (p=' '*4, d=k, id=T[k], rd=v))

        return modded_dims

    def _feed(self):
        """ Feed stub """
        try:
            self._feed_impl()
        except Exception as e:
            montblanc.log.exception("Feed exception")
            raise

    def _feed_impl(self):
        """ Implementation of queue feeding """
        S = self._tf_session
        DS = self._data_sources

        def feed_one_queue(queue, session, data_source, array_descriptor):
            feed_dict={
                ph: data_source[n][0](self, array_descriptor[n])
                for ph, n
                in itertools.izip(queue.placeholders, queue.fed_arrays)}

            total_bytes = np.sum(a.nbytes for a in feed_dict.itervalues())
            montblanc.log.info('Feeding {b} into {a}'.format(
                b=hcu.fmt_bytes(total_bytes),
                a=queue.fed_arrays))

            session.run(queue.placeholder_enqueue_op(), feed_dict=feed_dict)

        # Queues to be fed on each iteration
        iter_queues = tuple((self._input_queue,
            self._frequency_queue,
            self._uvw_queue,
            self._observation_queue,
            self._die_queue,
            self._dde_queue))

        # Iterate over time and baseline
        iter_dims = 'ntime', 'nbl'
        iter_strides = self.dim_local_size(*iter_dims)
        iter_args = zip(iter_dims, iter_strides)

        # Iterate through the hypercube space
        for i, d in enumerate(self.dim_iter(*iter_args, update_local_size=True)):
            self.update_dimensions(d)
            array_descriptors = self.arrays(reify=True)

            # Create a feed dictionary from all arrays in the feed queues
            feed_dict = { ph: DS[a][0](self, array_descriptors[a])
                for q in iter_queues
                for ph, a in zip(q.placeholders, q.fed_arrays) }

            montblanc.log.info("Enqueueing chunk {i} Start".format(i=i))

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            S.run([q.placeholder_enqueue_op() for q in iter_queues],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('feed-timeline.json', 'w') as f:
                f.write(ctf)

            montblanc.log.info("Enqueueing chunk {i} End".format(i=i))        

            # Now enqueue source chunks
            for ds in self.dim_iter(('npsrc', self.dim_local_size('npsrc')), update_local_size=True):
                self.update_dimensions(ds)

                montblanc.log.info("Enqueueing {s} point sources".format(s=ds[0]['local_size']))

                feed_dict = { ph: DS[a][0](self, array_descriptors[a])
                    for q in (self._point_source_queue,)
                    for ph, a in zip(q.placeholders, q.fed_arrays) }

                S.run(self._point_source_queue.placeholder_enqueue_op(),
                    feed_dict=feed_dict)

    def _compute_impl(self):
        """ Implementation of computation """
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        S = self._tf_session

        feed_dict = { ph: self.dim_global_size(n) for
            n, ph in self._src_ph_vars.iteritems() }

        feed_dict.update({ ph: getattr(self, n) for
            n, ph in self._property_ph_vars.iteritems() })

        result = S.run(self._tf_expr,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('compute-timeline.json', 'w') as f:
            f.write(ctf)

        return result

    def _compute(self):
        """ Compute stub """
        try:
            return self._compute_impl()
        except Exception as e:
            montblanc.log.exception("Compute exception")
            raise

    def _construct_tensorflow_expression(self):
        """ Constructs a tensorflow expression for computing the RIME """
        zero = tf.constant(0)

        # Pull RIME inputs out of the feed queues
        frequency, ref_frequency = self._frequency_queue.queue.dequeue()
        model_vis = self._input_queue.queue.dequeue()
        uvw, antenna1, antenna2 = self._uvw_queue.queue.dequeue()
        observed_vis, flag, weight = self._observation_queue.queue.dequeue()
        ebeam, antenna_scaling, point_errors = self._dde_queue.queue.dequeue()        
        gterm = self._die_queue.queue.dequeue()

        # Infer chunk dimensions
        nchan = tf.shape(frequency)[0]
        ntime = tf.shape(antenna1)[0]
        nbl = tf.shape(antenna1)[1]

        def point_cond(model_vis, npsrc):
            return tf.less(npsrc, self._src_ph_vars.npsrc)

        def point_body(model_vis, npsrc):
            lm, stokes, alpha = self._point_source_queue.queue.dequeue()
            # Source batch size
            nsrc = tf.shape(lm)[0]

            # Accumulate visiblities for this source batch
            cplx_phase = rime.phase(lm, uvw, frequency, CT=model_vis.dtype)
            bsqrt = rime.b_sqrt(stokes, alpha, frequency, ref_frequency)
            ejones = rime.e_beam(lm, point_errors, antenna_scaling, ebeam,
                self._property_ph_vars.parallactic_angle,
                self._property_ph_vars.beam_ll,
                self._property_ph_vars.beam_lm,
                self._property_ph_vars.beam_ul,
                self._property_ph_vars.beam_um)

            ant_jones = rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=lm.dtype)
            shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=lm.dtype)    
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                shape, ant_jones, flag, gterm, model_vis, False)

            return model_vis, npsrc + nsrc

        M, npsrc = tf.while_loop(point_cond, point_body, [model_vis, zero])

        return M

    def solve(self):
        f = self._feed_executor.submit(self._feed)
        c = self._compute_executor.submit(self._compute)

        done, not_done = cf.wait([f,c], return_when=cf.FIRST_COMPLETED)

        for d in done:
            data = d.result()
            print data
            print data.shape

    def close(self):
        self._feed_executor.shutdown()
        self._compute_executor.shutdown()
        self._tf_session.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()