#-------------------------------------------------------------------------------

class Priors(object):
    """
    Jonathan Zwart
    6 July 2013
    5 March 2014 - added general prior function

    Taken from MultiNEST priors.f90
    Usage e.g.:
        from priors import Priors
        pri=Priors()
        cube[0]=pri.UniformPrior(cube[0],1.0,199.0)
        cube[0]=pri.GeneralPrior(cube[0],'LOG',1.0,100.0)
        etc.
    """

    def __init__(self):
        pass

    def GeneralPrior(self,r,PriorType,x1,x2):
        if PriorType=='DELTA':
            return self.DeltaFunctionPrior(r,x1,x2)
        elif PriorType=='U':
            return self.UniformPrior(r,x1,x2)
        elif PriorType=='LOG':
            return self.LogPrior(r,x1,x2)
        elif PriorType=='GAUSS':
            return self.GaussianPrior(r,x1,x2)
        elif PriorType=='JEFF':
            return self.JeffreysPrior(r,x1,x2)
        else:
            print 'Unrecognised prior'
            return 1

    def DeltaFunctionPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Delta[x1]"""
        return x1

    def UniformPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Uniform[x1:x2]"""
        return x1+r*(x2-x1)

    def LogPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  LogUniform[x1:x2]"""
        from math import log10
        if (r <= 0.0):
                return -1.0e32
        else:
            lx1=log10(x1); lx2=log10(x2)
            return 10.0**(lx1+r*(lx2-lx1))

    def GaussianPrior(self,r,mu,sigma):
        """Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
        from math import sqrt
        from scipy.special import erfcinv
        if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
            return -1.0e32
        else:
            return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r))

    def JeffreysPrior(self,r,x1,x2):
        """Needs to be tested"""
        if (r <= 0.0):
                return -1.0e32
        else:
            #from math import log                                               
            from numpy import log
            if(x1==0):
                lx1 = sys.float_info.min
            else:
                lx1=log(x1)
            if(x2==0):
                lx2 = sys.float_info.min
            else:
                lx2=log(x2)
            return 1/(r*(lx2-lx1))

#-------------------------------------------------------------------------------
