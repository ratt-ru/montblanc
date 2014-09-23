import pyrap.tables as pt
import os
import montblanc

from montblanc.api.loaders import BaseLoader

ANTENNA_TABLE = 'ANTENNA'
SPECTRAL_WINDOW = 'SPECTRAL_WINDOW'

class MeasurementSetLoader(BaseLoader):
	def __init__(self, msfile):
		super(MeasurementSetLoader, self).__init__()

		self.tables = {}
		self.msfile = msfile
		self.antfile = os.path.join(self.msfile, ANTENNA_TABLE)
		self.freqfile = os.path.join(self.msfile, SPECTRAL_WINDOW)

		self.tables['main'] = tm \
			= pt.table(self.msfile, ack=False).query('ANTENNA1 != ANTENNA2')
		self.tables['ant']  = ta \
			= pt.table(self.antfile, ack=False)
		self.tables['freq'] = tf \
			= pt.table(self.freqfile, ack=False)

	def get_dims(self, auto_correlations=False):
		"""
		Returns a tuple with the number of timesteps, antenna and channels
		"""
		# Determine the problem dimensions
		na = self.tables['ant'].nrows()
		nbl = montblanc.nr_of_baselines(na, auto_correlations)
		nchan = self.tables['freq'].getcol('CHAN_FREQ').size
		ntime = self.tables['main'].nrows() // nbl

		return ntime, na, nchan

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		# Close all the tables
		for table in self.tables.itervalues():
			table.close()