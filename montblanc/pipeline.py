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

    def initialise(self, solver):
        """
        Iterates over each node in the pipeline,
        calling the initialise() method on each one.

        >>> slvr = Solver()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(slvr)
        >>> pipeline.execute(slvr)
        >>> pipeline.shutdown(slvr)
        """

        montblanc.log.debug('Initialising pipeline')

        try:
            for node in self.pipeline:
                montblanc.log.debug('Initialising node \'' + node.description() + '\'')
                node.initialise(solver)
                montblanc.log.debug('Done')
        except PipeLineError as e:
            montblanc.log.error('Pipeline Error occurred during RIME pipeline initialisation', exc_info=True)
            self.initialised = False
            return self.initialised
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred duri8\ng RIME pipeline initialisation', e
#            self.initialised = False
#            return self.initialised

        montblanc.log.debug('Initialisation of pipeline complete')

        self.initialised = True
        self.nr_of_executions = 0
        self.sum_execution_time = 0.0
        self.last_execution_time = 0.0
        self.pipeline_start = cuda.Event()
        self.pipeline_end = cuda.Event()        
        return self.initialised

    def execute(self, solver):
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
            if not self.initialise():
                return False

        try:
            # Record start of pipeline
            self.pipeline_start.record()

            for node in self.pipeline:

                montblanc.log.debug('Executing node \'' + node.description() + '\'')
                node.pre_execution(solver)
                montblanc.log.debug('pre')
                node.execute(solver)
                montblanc.log.debug('execute')
                node.post_execution(solver)
                montblanc.log.debug('post Done')

            # Record pipeline end
            self.pipeline_end.record()
            self.pipeline_end.synchronize()
            self.nr_of_executions += 1
            self.last_execution_time = self.pipeline_start.time_till(self.pipeline_end)
            self.sum_execution_time += self.last_execution_time

        except PipeLineError as e:
            montblanc.log.error('Pipeline Error occurred during RIME pipeline execution', exc_info=True)
            return False
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred during RIME pipeline execution', e
#            return False

        self.avg_execution_time = self.sum_execution_time / self.nr_of_executions

        montblanc.log.debug('Execution of pipeline complete')
        return False

    def shutdown(self, solver):
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
                    node.shutdown(solver)
                    montblanc.log.debug('Done')
            except PipeLineError as e:
                montblanc.log.error('Pipeline Error occurred during RIME pipeline shutdown', exc_info=True)
                success = False
#            except Exception, e:
#                print
#                print 'Unexpected exception occurred during RIME pipeline shutdown', e
#                success = False

        montblanc.log.debug('Shutdown of pipeline Complete')
        return success

    def __str__(self):
    	return ' '.join([node.description() for node in self.pipeline])