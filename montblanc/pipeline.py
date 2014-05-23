import argparse
import os.path
import sys

import montblanc

from montblanc.node import NullNode

class PipeLineError(Exception):
    """ Pipeline Exception base class """
    pass

class Pipeline:
    """ Class describing a pipeline of RIME equations """
    def __init__(self, node_list=None, user_options=None):
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

        if user_options is None:
            user_options = {}

        if type(user_options) is not dict:
            raise TypeError, 'user_options argument is not a dict'

        self.options = montblanc.default_pipeline_options()
        self.options.update(user_options)

        self.pipeline = node_list
        self.initialised = False

    def initialise(self, shared_data):
        """
        Iterates over each node in the pipeline,
        calling the initialise() method on each one.

        >>> sd = SharedData()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(sd)
        >>> pipeline.execute(sd)
        >>> pipeline.shutdown(sd)
        """

        print 'Initialising pipeline'

        try:
            for node in self.pipeline:
                print '\tInitialising node \'' + node.description() + '\'',
                node.initialise(shared_data)
                print 'Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline initialisation', e
            self.initialised = False
            return self.initialised
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred duri8\ng RIME pipeline initialisation', e
#            self.initialised = False
#            return self.initialised

        print 'Initialisation of pipeline complete'

        self.initialised = True
        return self.initialised

    def execute(self, shared_data):
        """
        Iterates over each node in the pipeline,
        calling the execute() method on each one.

        >>> sd = SharedData()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(sd)
        >>> pipeline.execute(sd)
        >>> pipeline.shutdown(sd)
        """

        print 'Executing pipeline'

        if not self.initialised:
            print '\t Pipeline was not initialised!'
            return False

        try:
            for node in self.pipeline:
                print '\tExecuting node \'' + node.description() + '\'',
                node.pre_execution(shared_data)
                print 'pre',
                node.execute(shared_data)
                print 'execute',
                node.post_execution(shared_data)
                print 'post Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline execution', e
            return False
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred during RIME pipeline execution', e
#            return False

        print 'Execution of pipeline complete'
        return False

    def shutdown(self, shared_data):
        """
        Iterates over each node in the pipeline,
        calling the shutdown() method on each one.

        >>> sd = SharedData()
        >>> pipeline = PipeLine([Node1(), Node2(), Node3()])
        >>> pipeline.initialise(sd)
        >>> pipeline.execute(sd)
        >>> pipeline.shutdown(sd)
        """

        print 'Shutting down pipeline'

        success = True

        # Even if shutdown fails, keep trying on the other nodes
        for node in self.pipeline:
            try:
                    print '\tShutting down node \'' + node.description() + '\'',
                    node.shutdown(shared_data)
                    print 'Done'
            except PipeLineError as e:
                print
                print 'Pipeline Error occurred during RIME pipeline shutdown', e
                success = False
#            except Exception, e:
#                print
#                print 'Unexpected exception occurred during RIME pipeline shutdown', e
#                success = False

        print 'Shutdown of pipeline Complete'
        return success

    def __str__(self):
    	return ' '.join([node.description() for node in self.pipeline])