#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Marzia Rivi
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

# Compare analytical chi squared gradient computation with respect to 
# Sersic parameters with the numerical gradent computation. 
 
# Input: filename - Measurement Set file
#        ns - number of sersic sources

import sys
import argparse
import math
import numpy as np
import montblanc
import montblanc.util as mbu

from  montblanc.config import RimeSolverConfig as Options

ARCS2RAD = np.pi/648000.

#scalelength in arcsec
minscale = 0.3*ARCS2RAD
maxscale = 3.5*ARCS2RAD

parser = argparse.ArgumentParser(description='TEST SERSIC GRADIENT')
parser.add_argument('msfile', help='Input MS filename')
parser.add_argument('-ns',dest='nssrc', type=int, default=1, help='Number of Sersic Galaxies')

args = parser.parse_args(sys.argv[1:])

# Get the RIME solver
# Enable gradient computation: sersic_gradient=True
slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=0, gaussian=0, sersic=args.nssrc),
        init_weights=None, weight_vector=False,  
        sersic_gradient=True, dtype='double', version='v4')

with montblanc.rime_solver(slvr_cfg) as slvr:

    nsrc, nssrc, ntime, nchan = slvr.dim_local_size('nsrc', 'nssrc', 'ntime', 'nchan')
    
# Random source coordinates in the l,m (brightness image) domain
    l= slvr.ft(np.random.random(nsrc)*100*ARCS2RAD)
    m= slvr.ft(np.random.random(nsrc)*100*ARCS2RAD) 
    lm=mbu.shape_list([l,m], shape=slvr.lm.shape, dtype=slvr.lm.dtype)
    slvr.transfer_lm(lm)

# Brightness matrix for sources
    stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
    I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
    I[:] = np.ones(shape=I.shape)*70.
    Q[:] = np.zeros(shape=Q.shape)
    U[:] = np.zeros(shape=U.shape)
    V[:] = np.zeros(shape=V.shape)
    slvr.transfer_stokes(stokes)
    alpha = slvr.ft(np.ones(nssrc*ntime)*(-0.7)).reshape(nsrc,ntime)
    slvr.transfer_alpha(alpha)
    
    # If there are sersic sources, create their
    # shape matrix and transfer it.
    mod = slvr.ft(np.random.random(nssrc))*0.3
    angle = slvr.ft(np.random.random(nssrc))*2*np.pi
    e1 = mod*np.sin(angle)
    e2 = mod*np.cos(angle)
    R = slvr.ft(np.random.random(nssrc)*(maxscale-minscale))+minscale
    sersic_shape = slvr.ft(np.array([e1,e2,R])).reshape((3,nssrc))
    slvr.transfer_sersic_shape(sersic_shape)
    print sersic_shape

    #Set visibility noise variance (muJy)
    time_acc = 60
    efficiency = 0.9
    channel_bandwidth_hz = 20e6
    SEFD = 400e6
    sigma = (SEFD*SEFD)/(2.*time_acc*channel_bandwidth_hz*efficiency*efficiency)
    slvr.set_sigma_sqrd(sigma)


    # Create observed data and upload it to the GPU
    slvr.solve()
    with slvr.context:
        observed = slvr.retrieve_model_vis()

    noiseR = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noiseI = np.random.normal(0,np.sqrt(sigma),size=observed.shape)
    noise = noiseR +1j*noiseI
    observed = observed+noise
    print 'transfer to GPU'
    slvr.transfer_observed_vis(observed)
 
    print slvr

    slvr.solve() 
    dq_gpu = slvr.X2_grad  
    print "analytical GPU: ",dq_gpu

    # Execute the pipeline
    slvr.solve()
    l1 = slvr.X2
    dq = np.empty([3,nssrc])
    for i in xrange(nssrc):
        e1_inc = sersic_shape.copy()
        e1_inc[0,i] = e1_inc[0,i]+0.000001
        slvr.transfer_sersic_shape(e1_inc)
        slvr.solve()
        l2 = slvr.X2
        dq[0,i] = (l2-l1)*1e6
        e2_inc = sersic_shape.copy()
        e2_inc[1,i] = e2_inc[1,i]+ 0.000001
        slvr.transfer_sersic_shape(e2_inc)
        slvr.solve()
        l2 = slvr.X2
        dq[1,i] = (l2-l1)*1e6
        R_inc = sersic_shape.copy()
        R_inc[2,i] = R_inc[2,i]+0.00000001
        slvr.transfer_sersic_shape(R_inc)
        slvr.solve()
        l2 = slvr.X2
        dq[2,i] = (l2-l1)*1e8
    print "numeric",dq


