import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

from montblanc.node import Node

class RimeChiSquaredReduce(Node):
    """
    Encapsulates GPU code for computing the reduction of
    the values within the sum of a Chi Squared computation.
    """
    def __init__(self, noise_vector=False):
        """
        Parameters:
        -----------
        noise_vector : boolean
            True if each value within the sum of a Chi Squared should
            be individually divided by a corresponding sigma squared
            in the noise vector, and then summed.

            False if each value within the sum of a Chi Squared should
            be summed, and the final result divided by a single
            sigma squared value.
        """
        super(RimeChiSquaredReduce, self).__init__()
        self.noise_vector = noise_vector

    def initialise(self, shared_data):
        # OK, we need to handle noise vectors
        if self.noise_vector:
            sd = shared_data
            # Cater for single and double precision arguments
            args = 'float * x, float * y' if sd.is_float() else \
                'double * x, double * y'

            # Create a reduction kernel. map_expr is a
            # map expression that will compute the division
            # of the model/data diff by the appropriate
            # sigma squared value in the noise vector.
            # reduce_expr expresses the sum of two map_expr
            self.kernel = ReductionKernel(sd.ft, neutral='0',
                reduce_expr='a+b', map_expr='x[i]/y[i]',
                arguments=args)

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def execute(self, shared_data):
        sd = shared_data
        
        # If we're not catering for a noise vector,
        # call the simple reduction and divide by sigma squared.
        # Otherwise, call the more complicated reduction kernel that
        # internally divides by the noise vector
        if not self.noise_vector:
            sd.set_X2(gpuarray.sum(sd.chi_sqrd_result_gpu).get()/sd.sigma_sqrd)
        else:
            sd.set_X2(self.kernel(sd.chi_sqrd_result_gpu, sd.noise_vector_gpu).get())

    def post_execution(self, shared_data):
        pass