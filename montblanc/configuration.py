from ruamel.yaml import YAML

def load_config(file):
    yaml = YAML()
    return yaml.load(file)

def config_validator():
    from cerberus import Validator
    from montblanc.src_types import default_sources

    schema = {
        'data_source': {
            'type': 'string',
            'allowed': ['default', 'test'],
            'default': 'default' },

        'dtype': {
            'type': 'string',
            'allowed': ['float', 'double'],
            'default': 'double'},

        'auto_correlations': {
            'type': 'boolean',
            'default': False },

        'polarisation_type': {
            'type': 'string',
            'allowed': ['linear', 'circular'],
            'default': 'linear' },

        'mem_budget': {
            'type': 'integer',
            'min': 1024,
            'default': 1024*1024*1024 },

        'source_batch_size': {
            'type': 'integer',
            'min': 0,
            'default': 500 },

        # TODO: Remove sources
        'sources': {
            'type': 'dict',
            'default': default_sources() }
    }

    return Validator(schema)
