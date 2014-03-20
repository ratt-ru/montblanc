import numpy as np

na = 6               # nr of antennas
nbl = (na**2 + na)/2 # nr of baselines

for BL in range(nbl):
	ANT1 = int(np.floor((np.sqrt(1+8*BL)-1)/2))
	ANT2 = BL - (ANT1**2+ANT1)/2
	ANT1 += 1

	print 'Baseline %d == (%d, %d)' % (BL, ANT1, ANT2)