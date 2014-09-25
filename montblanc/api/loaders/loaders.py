class BaseLoader(object):
	def __init__(self):
		pass

	def load(self, solver, **kwargs):
		raise NotImplementedError

	def __enter__(self):
		raise NotImplementedError

	def __exit__(self, type, value, traceback):
		raise NotImplementedError