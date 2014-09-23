from montblanc.node import Node

import montblanc.ext.crimes

class RimeJonesReduce(Node):
    def __init__(self):
        super(RimeJonesReduce, self).__init__()
    def initialise(self, solver):
        pass
    def shutdown(self, solver):
        pass
    def pre_execution(self, solver):
        pass
    def execute(self, solver):
        slvr = solver

        if slvr.is_float():
            montblanc.ext.crimes.segmented_reduce_complex64_sum(
                data=slvr.jones_gpu, seg_starts=slvr.keys_gpu,
                seg_sums=slvr.vis_gpu, cc=slvr.cc, device_id=0)
        else:
            montblanc.ext.crimes.segmented_reduce_complex128_sum(
                data=slvr.jones_gpu, seg_starts=slvr.keys_gpu,
                seg_sums=slvr.vis_gpu, cc=slvr.cc, device_id=0)

    def post_execution(self, solver):
        pass