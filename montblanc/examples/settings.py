"""
Parameter file for bayesnest.py
"""

import numpy
import time
import os
import sys
import scipy.constants as sc
import pyrap.tables as pt
import pymultinest
import numpy as np
from mpi4py import MPI
from math import sqrt
import logging

#-------------------------------------------------------------------------------
# define some constants

deg2rad = sc.pi / 180.0;
arcsec2rad = sc.pi / 180.0 / 3600.0;
sqrtTwo=sqrt(2.0)

#------------------------------------------------------------------------------
# Command-line options

sigmaSim=None # Error on each visibility that goes into the predictions - should be the same as SIMULATION NOISE; None -> fit it

# Montblanc settings
loggingLevel=logging.WARN                      # Logging level
msfile='gaussian-off-centre-20x16at45/WSRT.MS' # Input MS file
store_cpu=False         # Carry out the calculation on the CPU
use_noise_vector=False                         # Varying noise level
dtype=np.float32                               # or np.float64
npsrc=0                                        # no. point sources
ngsrc=1                                        # no. gaussians

# Multinest settings
hypo=51 # hypothesis id - 0/1/2/3
verbose=True # Helpful print statements in the output
threads=2 # No. of meqserver threads to use
nlive=1000 # Number of live points for MultiNest
#from math import log
evtol=0.5 # Evidence tolerance for MultiNest
#evtol=log(3.0)
efr=0.1   # Target sampling efficiency
resume=False # Resume interrrupted MultiNest runs
seed=4747# Random no. generator seed (-ve for system clock)
ins=False # Use Importance Nested Sampling? (Multinest)
maxiter=0 # maximum number of iterations for multinest


#------------------------------------------------------------------------------
# Some options to multinest that depend on the value of hypo

# Model 0 (noise only) -- 3 params (all = 0.0)
# Model 1 (noise + source 1 + source 2) -- distinct position priors
# Model 2 (noise + source 3 [gaussian])
# Model 3 (noise + source 1 [single atom] )

n_params=8
multimodal=False

mode_tolerance=-1e90 # THIS IS AN ESSENTIAL BUGFIX TO ALLOW MULTI MODES

#------------------------------------------------------------------------------

# Set up the parameters for plotting

'''parameters=['C','alpha','Smin','Smax']
"""plotRanges={'C':[0.0,100.0],
            'alpha':[-2.5,-0.1],
            'Smin':[0.0,5.0],
            'Smax':[5.0,100.0]}"""
plotTruth={'S':trueS,
          'l':0.0,
          'm':0.0,
          'lproj':truelproj,
          'mproj':truemproj,
          'rat':ratio_true,
          'maj':truemaj,
          'min':truemin,
          'pa':truepa}
'''
#-------------------------------------------------------------------------------

# Priors

e1min = 0.0 * arcsec2rad; e1max = 20.0 * arcsec2rad;
e2min = -20.0 * arcsec2rad; e2max = 20.0 * arcsec2rad;
dxmin=200.0 * arcsec2rad; dxmax=+250.0 * arcsec2rad;
dymin=335.0 * arcsec2rad; dymax=+385.0 * arcsec2rad;
dy1min=350.0 * arcsec2rad; dy1max=+360.0 * arcsec2rad;
dy2min=360.0 * arcsec2rad; dy2max=+370.0 * arcsec2rad;
Smin=0.0; Smax=2.0 # Jy

# For the Gaussian case (hypo 2), the parametrization is as follows:
# emaj (e1) * sin(p.a)
# emaj (e1) * cos(p.a)
# emin (e2) / emaj (e1)

#-------------------------------------------------------------------------------


