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

class Node(object):
    """
    Abstract pipeline node class.

    This class should be extended to provide a concrete implementation of a pipeline node class.

    >>> class GPUNode(Node):
    >>>     def __init__(self):
    >>>         super(type(self), self).__init__()
    >>>     def pre_execution(self, solver):
    >>>         solver.N = 1024
    >>>         solver.a_cpu = np.random.random(solver.N)
    >>>         solver.a_gpu = gpuarray.to_gpu(solver.a_cpu)
    >>>     def post_execution(self, solver):
    >>>         del shared.data.N
    >>>         del solver.a_gpu
    >>>         del solver.a_cpu
    >>>     def execute(self, solver):
    >>>         solver.result = (solver.a_gpu*2./3.).get()
    """

    def __init__(self):
        pass

    def description(self):
        """ Returns a string description of the node """
        #raise NotImplementedError, self.not_implemented_string(type(self).description.__name__)
        return self.__class__.__name__

    def initialise(self, solver, stream=None):
        """
        Abstract method that should be overriden by derived classes.
        This method is called before the execution of the pipeline. Any initialisation
        required by the node should be performed here.

        Arguments:
            solver - object derived from BaseSolver
                Contains data shared amongst pipeline components
            stream - cuda stream
                Asynchronous stream to execute this node on. Can be None
                indicating asynchronous transfers should not take place.
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).initialise.__name__)

    def shutdown(self, solver, stream=None):
        """
        Abstract method that should be overriden by derived classes.
        This method is called after the execution of the pipeline. Any cleanup required
        by the node should be performed here.

        Arguments:
            solver - object derived from BaseSolver
                Contains data shared amongst pipeline components
            stream - cuda stream
                Asynchronous stream to execute this node on. Can be None
                indicating asynchronous transfers should not take place.
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).shutdown.__name__)

    def execute(self, solver, stream=None):
        """
        Abstract method that should be overriden by derived classes.
        The derived class should implement the code that the node should
        execute. Most likely this will be a PyCUDA GPU kernel of some sort,
        but the pipeline model is general enough to support CPU calls etc.

        Arguments:
            solver - object derived from BaseSolver
                Contains data shared amongst pipeline components
            stream - cuda stream
                Asynchronous stream to execute this node on. Can be None
                indicating asynchronous transfers should not take place.
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).execute.__name__)

    def pre_execution(self, solver, stream=None):
        """
        Hook for writing code that will be executed prior
        to the execute() method.


        Arguments:
            solver - object derived from BaseSolver
                Contains data shared amongst pipeline components
            stream - cuda stream
                Asynchronous stream to execute this node on. Can be None
                indicating asynchronous transfers should not take place.
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).pre_execution.__name__)

    def post_execution(self, solver, stream=None):
        """
        Hook for writing code that will be executed after
        the execute() method.

        Arguments:
            solver - object derived from BaseSolver
                Contains data shared amongst pipeline components
            stream - cuda stream
                Asynchronous stream to execute this node on. Can be None
                indicating asynchronous transfers should not take place.
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).post_execution.__name__)

    def __not_implemented_string(self, function_name):
        return ' '.join(['method', function_name, 'not implemented in class',type(self).__name__,'derived from abstract class', Node.__name__])

class NullNode(Node):
    def __init__(self): super(NullNode, self).__init__()
    def initialise(self, solver, stream=None):
        pass
    def shutdown(self, solver, stream=None):
        pass
    def pre_execution(self, solver, stream=None):
        pass
    def execute(self, solver, stream=None):
        pass
    def post_execution(self, solver, stream=None):
        pass
