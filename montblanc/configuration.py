from ruamel.yaml import YAML

def load_config(file):
    yaml = YAML()
    return yaml.load(file)

def config_validator():
    from cerberus import Validator
    from montblanc.src_types import default_sources

    class DescriptionValidator(Validator):
        def _validate___description__(self, __description__,
                                                field, value):
            """
            Dummy rule so that '__description__' keys
            can be placed in the schema

            The rule's arguments are validated against this schema:
            {'type': 'string'}
            """
            pass

    schema = {
        'data_source': {
            'type': 'string',
            'allowed': ['default', 'test'],
            'default': 'default',
            '__description__': "Data Source for initialising inputs. "
                               "If 'default', initialised with defaults, "
                               "If 'test', initialised with sensible "
                               "test data. " },

        'dtype': {
            'type': 'string',
            'allowed': ['float', 'double'],
            'default': 'double',
            '__description__': "Floating Point precision of solution" },

        'auto_correlations': {
            'type': 'boolean',
            'default': False,
            '__description__': "Take auto-correlations into account "
                               "when number of baselines "
                               "from number of antenna." },

        'polarisation_type': {
            'type': 'string',
            'allowed': ['linear', 'circular'],
            'default': 'linear',
            '__description__': "Type of polarisation. "
                               "Can be 'linear' or 'circular'." },

        'mem_budget': {
            'type': 'integer',
            'min': 1024,
            'default': 1024*1024*1024,
            '__description__': "Memory budget for solving a single "
                               "tile of the problem on a CPU/GPU." },

        'source_batch_size': {
            'type': 'integer',
            'min': 0,
            'default': 500 },

        'version': {
            'type': 'string',
            'default': 'tf' },
    }

    return DescriptionValidator(schema)
