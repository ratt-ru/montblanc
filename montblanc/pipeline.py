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

import argparse
import os.path
import sys

import pycuda.driver as cuda

import montblanc

from montblanc.node import NullNode

class PipeLineError(Exception):
    """ Pipeline Exception base class """
    pass

class Pipeline:
    """ Class describing a pipeline of RIME equations """
    def __init__(self, node_list=None):
        """ Initialise the pipeline with a list of nodes

        Keyword arguments:
            node_list -- A list of nodes defining the pipeline

        >>> pipeline = PipeRimes([InitNode(), \\
            ProcessingNode1(), ProcessingNode2(), CleanupNode()])
        """
        if node_list is None:
        	node_list = [NullNode()]

        if type(node_list) is not list:
            raise TypeError, 'node_list argument is not a list'

        self.pipeline = node_list
        self.initialised = False

    def is_initialised(self):
        return self.initialised

    def initialise(self, solver, stream=None):
        """
        Iterates over each node in the pipeline,
        calling the initialise() method on each one.

        >>> slvr = Solver()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(slvr)
        >>> pipeline.execute(slvr)
        >>> pipeline.shutdown(slvr)
        """

        if self.is_initialised():
            return self.is_initialised()

        montblanc.log.debug('Initialising pipeline')

        try:
            for node in self.pipeline:
                montblanc.log.debug('Initialising node \'' + node.description() + '\'')
                node.initialise(solver, stream)
                montblanc.log.debug('Done')
        except PipeLineError as e:
            montblanc.log.error('Pipeline Error occurred during RIME pipeline initialisation', exc_info=True)
            self.initialised = False
            return self.initialised
        except Exception as e:
            montblanc.log.error(('Unexpected exception occurred '
                'during RIME pipeline initialisation'), exc_info=True)
            self.initialised = False
            return self.is_initialised()

        montblanc.log.debug('Initialisation of pipeline complete')

        self.initialised = True
        self.nr_of_executions = 0
        return self.is_initialised()

    def execute(self, solver, stream=None):
        """
        Iterates over each node in the pipeline,
        calling the execute() method on each one.

        >>> slvr = Solver()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(slvr)
        >>> pipeline.execute(slvr)
        >>> pipeline.shutdown(slvr)
        """

        montblanc.log.debug('Executing pipeline')

        if not self.is_initialised():
            montblanc.log.error('Pipeline was not initialised!')
            if not self.initialise(solver, stream):
                return False

        try:
            # Record start of pipeline

            for node in self.pipeline:

                montblanc.log.debug('Executing node \'' + node.description() + '\'')
                node.pre_execution(solver, stream)
                montblanc.log.debug('pre')
                node.execute(solver, stream)
                montblanc.log.debug('execute')
                node.post_execution(solver, stream)
                montblanc.log.debug('post Done')

            # Record pipeline end
            self.nr_of_executions += 1

        except PipeLineError as e:
            montblanc.log.error('Pipeline Error occurred during RIME pipeline execution', exc_info=True)
            return False
        except Exception as e:
            montblanc.log.error(('Unexpected exception occurred '
                'during RIME pipeline execution'), exc_info=True)
            return False

        montblanc.log.debug('Execution of pipeline complete')
        return False

    def shutdown(self, solver, stream=None):
        """
        Iterates over each node in the pipeline,
        calling the shutdown() method on each one.

        >>> slvr = Solver()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(slvr)
        >>> pipeline.execute(slvr)
        >>> pipeline.shutdown(slvr)
        """

        montblanc.log.debug('Shutting down pipeline')

        success = True

        # Even if shutdown fails, keep trying on the other nodes
        for node in self.pipeline:
            try:
                montblanc.log.debug('Shutting down node \'' + node.description() + '\'')
                node.shutdown(solver, stream)
                montblanc.log.debug('Done')
            except PipeLineError as e:
                montblanc.log.error(('Pipeline Error occurred '
                    'during RIME pipeline shutdown'), exc_info=True)
                success = False
            except Exception as e:
                montblanc.log.error(('Unexpected exception occurred '
                    'during RIME pipeline shutdown'), exc_info=True)
                success = False

        montblanc.log.debug('Shutdown of pipeline Complete')
        # The pipeline is no longer active
        self.initialised = False

        return success

    def __str__(self):
    	return ' '.join([node.description() for node in self.pipeline])