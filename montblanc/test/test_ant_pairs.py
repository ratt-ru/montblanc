import numpy as np
from montblanc.impl.biro.v1.BiroSolver import BiroSolver as BSD1
from montblanc.impl.biro.v2.BiroSolver import BiroSolver as BSD2

class Problem: pass

p1, p2, p3 = Problem(), Problem(), Problem()

na, ntime, nchan, npsrc, ngsrc = 5, 2, 4, 1, 1

p1.slvr = BSD1(na=na,ntime=ntime,nchan=nchan,npsrc=npsrc,ngsrc=ngsrc)
p2.slvr = BSD2(na=na,ntime=ntime,nchan=nchan,npsrc=npsrc,ngsrc=ngsrc)
p3.slvr = BSD2(na=na,ntime=ntime,nchan=nchan,npsrc=npsrc,ngsrc=ngsrc)

nbl, nsrc = p1.slvr.nbl, p1.slvr.nsrc

p1.ap = p1.slvr.get_default_ant_pairs()
p2.ap = p2.slvr.get_default_ant_pairs()
p3.ap = p3.slvr.get_default_ant_pairs()

print 'p1.ap[0]', p1.ap[0]
print 'p1.ap[1]', p1.ap[1]
print 'p1.ap.shape', p1.ap.shape

print p1.ap[0].reshape(nbl*ntime)
print p1.ap[0].reshape(nbl*ntime)*ntime
print p1.ap[0].reshape(nbl*ntime)*ntime + np.tile(np.arange(ntime),nbl)
print p1.ap[1].reshape(nbl*ntime)
print p1.ap[1].reshape(nbl*ntime)*ntime
print p1.ap[1].reshape(nbl*ntime)*ntime + np.tile(np.arange(ntime),nbl)

cs = nchan*nsrc
tcs = ntime*cs

ant1 = np.repeat(p1.ap[0],cs)*tcs + np.tile(np.arange(tcs), nbl)
ant2 = np.repeat(p1.ap[1],cs)*tcs + np.tile(np.arange(tcs), nbl)

ant1_check = np.empty(shape=(nbl,ntime,nchan,nsrc), dtype=np.int32)

for bl in range(nbl):
	for t in range(ntime):
		for ch in range(nchan):
			for src  in range(nsrc):
				ant = p1.ap[0,bl,t]
				value = (ant*ntime*nchan + t*nchan + ch)*nsrc + src
				#print bl, t, ch, src, ant, value
				ant1_check[bl,t,ch,src] =  value


p1apflat = p1.ap[0].reshape(nbl*ntime)*ntime + np.tile(np.arange(ntime),nbl)
idx = np.ix_(p1apflat,np.arange(nchan), np.arange(nsrc))
ravel_idx = np.ravel_multi_index(idx,(nbl*ntime,nchan,nsrc)).flatten()

data = np.random.random(size=(ntime,na,nchan,nsrc)).flatten()
d = (ravel_idx == ant1)
print 'diff', d
print 'ant1', ant1
print 'ant1_check', ant1_check.flatten()
print 'ravel_idx', ravel_idx
print 'data[ant1].shape', data[ant1].shape
print 'data[ravel_idx].shape', data[ravel_idx].shape

assert (ant1.flatten() == ant1_check.flatten()).all()


print 'p2.ap[0]', p2.ap[0]
print 'p2.ap[1]', p2.ap[1]
print 'p2.ap.shape', p2.ap.shape

print p2.ap[0].reshape(nbl*ntime)
print p2.ap[0].reshape(nbl*ntime)*nbl + np.tile(np.arange(nbl),ntime)
print p2.ap[0].reshape(nbl*ntime)*nbl + np.tile(np.arange(nbl),ntime)
print p2.ap[1].reshape(nbl*ntime)
print p2.ap[1].reshape(nbl*ntime)*nbl + np.tile(np.arange(nbl),ntime)
print p2.ap[1].reshape(nbl*ntime)*nbl + np.tile(np.arange(nbl),ntime)

ant1 = np.empty(shape=(ntime,nbl,nsrc,nchan), dtype=np.int32)

for t in range(ntime):
	for bl in range(nbl):
		for src  in range(nsrc):
			for ch in range(nchan):
				ant = p2.ap[0,t,bl]
				value = (t*na*nsrc + ant*nsrc + src)*nchan + ch
				ant1[t,bl,src,ch] =  value


data = np.random.random(size=(ntime,na,nsrc,nchan)).flatten()
print 'data.shape', data.shape

ant1 = ant1.flatten()
print 'ant1.shape %s ant1\n%s' %  (ant1.shape, ant1)

p2apflat = p2.ap[0].reshape(nbl*ntime)
print 'p2apflat %s\n%s' % (p2apflat.shape, p2apflat)
tmp = p2apflat*nchan*nsrc
print 'p2apflat %s\n%s' % (tmp.shape, tmp)
tmp = tmp + np.repeat(np.arange(ntime),nbl)*nchan*nsrc*na
print 'p2apflat %s\n%s' % (tmp.shape, tmp)
tmp = np.repeat(tmp, nsrc*nchan)
print 'p2apflat %s\n%s' % (tmp.shape, tmp)
#tmp = tmp + np.tile(np.arange(nsrc*nchan),nbl*ntime) #+ np.repeat(np.arange(ntime),nbl*nsrc*nchan)*ntime
tmp = tmp + np.tile(np.arange(nsrc*nchan),nbl*ntime)
print 'p2apflat %s\n%s' % (tmp.shape, tmp)

print 'ant1.shape %s ant1\n%s' %  (ant1.shape, ant1)
print 'diff', tmp - ant1

assert (ant1 == tmp).all()

print 'data[ant1].shape', data[ant1].shape
print 'data[tmp].shape', data[tmp].shape

print 'dims: ntime: %d nbl: %d nsrc: %d nchan %d ntime*nbl %d, ntime*nbl*nsrc*nchan %d' % \
	(ntime, nbl, nsrc, nchan, ntime*nbl, ntime*nbl*nsrc*nchan)

p2apflat = p2.ap[0].reshape(nbl*ntime)
print 'p2apflat %s\n%s' % (p2apflat.shape, p2apflat)
tmp = p2apflat + np.repeat(np.arange(ntime),nbl)*nbl
print 'p2apflat %s\n%s' % (tmp.shape, tmp)

p2apflat = p2.ap[0].reshape(nbl*ntime)
print 'p2apflat %s\n%s' % (p2apflat.shape, p2apflat)
tmp = p2apflat + np.repeat(np.arange(ntime)*nbl,nbl)
print 'p2apflat %s\n%s' % (tmp.shape, tmp)



tmp1 = p1.ap.reshape(2,ntime*nbl)
print 'p1 ap\n%s' % tmp1
tmp1 = tmp1*ntime + np.tile(np.arange(ntime),nbl)
print 'p1 ap\n%s' % tmp1

ant2 = np.empty(shape=(ntime,nbl),dtype=np.int32)

for t in range(ntime):
	for bl in range(nbl):
		ant = p2.ap[0,t,bl]
		ant2[t,bl] = t*na + ant

tmp2 = p2.ap.reshape(2,ntime*nbl)
print 'p2 ap\n%s' % tmp2
tmp2 = tmp2 + np.repeat(np.arange(ntime),nbl)*na
print 'p2 ap\n%s' % tmp2
print 'ant2\n%s' % ant2.flatten()
