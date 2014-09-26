import numpy as np

def cpu_name(name):
    """ Constructs a name for the CPU version of the array """
    return name + '_cpu'

def gpu_name(name):
    """ Constructs a name for the GPU version of the array """
    return name + '_gpu'

def transfer_method_name(name):
    """ Constructs a transfer method name, given the array name """
    return 'transfer_' + name

def shape_name(name):
    """ Constructs a name for the array shape member, based on the array name """
    return name + '_shape'

def dtype_name(name):
    """ Constructs a name for the array data-type member, based on the array name """
    return name + '_dtype'

def setter_name(name):
    """ Constructs a name for the property, based on the property name """
    return 'set_' + name

def fmt_array_line(name,size,dtype,cpu,gpu,shape):
    """ Format array parameters on an 80 character width line """
    return '%-*s%-*s%-*s%-*s%-*s%-*s' % (
        20,name,
        10,size,
        15,dtype,
        4,cpu,
        4,gpu,
        20,shape)

def fmt_property_line(name,dtype,value,default):
    return '%-*s%-*s%-*s%-*s' % (
        20,name,
        10,dtype,
        20,value,
        20,default)

def fmt_bytes(nbytes):
    """ Returns a human readable string, given the number of bytes """
    for x in ['B','KB','MB','GB']:
        if nbytes < 1024.0:
            return "%3.1f%s" % (nbytes, x)
        nbytes /= 1024.0
    
    return "%.1f%s" % (nbytes, 'TB')

def array_bytes(shape, dtype):
    """ Estimates the memory in bytes required for an array of the supplied shape and dtype """
    return np.product(shape)*np.dtype(dtype).itemsize