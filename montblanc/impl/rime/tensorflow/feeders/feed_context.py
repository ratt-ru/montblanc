def _get_public_methods(obj):
	return set(method for method in dir(obj)
		if callable(getattr(obj, method))
		and not method.startswith('_'))

class _setter_property(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__
    
    def __set__(self, obj, value):
        return self.func(obj, value)

class FeedContext(object):
	"""
	Context for queue arrays.

	Proxies methods of a hypercube and provides access to configuration
	"""
	__slots__ = ('_cube', '_cfg', '_name', '_shape', '_dtype',
		'_cube_methods')

	def __init__(self, name, cube, slvr_cfg, shape, dtype):
		self._name = name
		self._cube = cube
		self._cfg = slvr_cfg
		self._shape = shape
		self._dtype = dtype
		self._cube_methods = _get_public_methods(cube)

		# Fall over if there's any intersection between the
		# public methods on the hypercube, the current class
		# and the cfg
		intersect = set.intersection(self._cube_methods,
			_feed_context_methods,
			_get_public_methods(slvr_cfg))

		if len(intersect) > 0:
			raise ValueError("'{i}' methods intersected on context"
				.format(i=intersect))

	@_setter_property
	def cube(self, value):
		self._cube = value

	@property
	def cfg(self):
		return self._cfg
		
	@cfg.setter
	def cfg(self, value):
		self._cfg = value

	@property
	def shape(self):
		return self._shape

	@shape.setter
	def shape(self, value):
		self._shape = value

	@property
	def dtype(self):
		return self._dtype

	@dtype.setter
	def dtype(self, value):
		self._dtype = value

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, value):
		self._name = value

	def __getattr__(self, name):
		# Defer to the hypercube
		if name in self._cube_methods:
			return getattr(self._cube, name)
		# Avoid recursive calls to getattr
		elif hasattr(self, name):
			return getattr(self, name)
		else:
			raise AttributeError(name)

_feed_context_methods = _get_public_methods(FeedContext)
