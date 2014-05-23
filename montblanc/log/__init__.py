import montblanc

import logging.config

logging.config.fileConfig()

"""
def get_file_handler(level, filename=None):
	if filename is None:
		filename = 'montblanc.log'

	handler = logging.FileHandler(filename)
	handler.setLevel(level)

# Default log level of warn
level = logging.WARN

log = logging.getLogger('montblanc')
log.setLevel(level)

file_handler = logging.FileHandler('montblanc.log')
file_handler.set_level(level)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# add the handlers to the logger

log.addHandler(file_handler)
"""