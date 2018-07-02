from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Dataset(object):
    """
    Abstract Dataset object
    """
    def dim_sizes(self):
        raise NotImplementedError()

    def dim_chunks(self):
        raise NotImplementedError()

    def dataset(self, chunks=None):
        raise NotImplementedError()
