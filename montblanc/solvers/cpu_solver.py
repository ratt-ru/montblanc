from rime_solver import RIMESolver

class CPUArrayDescriptor(object):
    """ Descriptor class for NumPy ndarrays arrays on the CPU """
    def __init__(self, record_key, default=None):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        instance.check_array(self.record_key, value)
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class CPUSolver(RIMESolver):
    def __init__(self, slvr_cfg):
        super(CPUSolver, self).__init__(slvr_cfg)        