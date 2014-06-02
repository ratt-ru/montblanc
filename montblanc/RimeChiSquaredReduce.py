import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from montblanc.node import Node

""" Simplest kernel ever! """
class RimeChiSquaredReduce(Node):
    def __init__(self):
        super(RimeChiSquaredReduce, self).__init__()

    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def execute(self, shared_data):
        sd = shared_data
        sd.set_X2(gpuarray.sum(sd.chi_sqrd_result_gpu).get()/sd.sigma_sqrd)

    def post_execution(self, shared_data):
        pass