from functools import partial
import textwrap

from ruamel.yaml import YAML

def load_config(file):
    yaml = YAML()
    return yaml.load(file)

def raise_validator_errors(validator):
    """
    Raise any errors associated with the validator.

    Parameters
    ----------
    validator : :class:`cerberus.Validator`
        Validator

    Raises
    ------
    ValueError
        Raised if errors existed on `validator`.
        Message describing each error and information
        associated with the configuration option
        causing the error.
    """

    if len(validator._errors) == 0:
        return

    def _path_str(path, name=None):
        """ String of the document/schema path. `cfg["foo"]["bar"]` """
        L = [name] if name is not None else []
        L.extend('["%s"]' % p for p in path)
        return "".join(L)

    def _path_leaf(path, dicts):
        """ Dictionary Leaf of the schema/document given the path """
        for p in path:
            dicts = dicts[p]

        return dicts

    wrap = partial(textwrap.wrap, initial_indent=' '*4,
                                subsequent_indent=' '*8)

    msg = ["There were configuration errors:"]

    for e in validator._errors:
        schema_leaf = _path_leaf(e.document_path, validator.schema)
        doc_str = _path_str(e.document_path, "cfg")

        msg.append("Invalid configuration option %s == '%s'." % (doc_str, e.value))

        try:
            otype = schema_leaf["type"]
            msg.extend(wrap("Type must be '%s'." % otype))
        except KeyError:
            pass

        try:
            allowed = schema_leaf["allowed"]
            msg.extend(wrap("Allowed values are '%s'." % allowed))
        except KeyError:
            pass

        try:
            description = schema_leaf["__description__"]
            msg.extend(wrap("Description: %s" % description))
        except KeyError:
            pass

    raise ValueError("\n".join(msg))


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
                               "test data." },

        'device_type': {
            'type': 'string',
            'allowed': ['CPU', 'GPU'],
            'default': 'GPU',
            '__description__': "Default compute device." },

        'dtype': {
            'type': 'string',
            'allowed': ['float', 'double'],
            'default': 'double',
            '__description__': "Floating Point precision of "
                                "inputs and solutions." },

        'auto_correlations': {
            'type': 'boolean',
            'default': False,
            '__description__': "Take auto-correlations into account "
                               "when computing number of baselines "
                               "from number of antenna." },

        'polarisation_type': {
            'type': 'string',
            'allowed': ['linear', 'circular'],
            'default': 'linear',
            '__description__': "Type of polarisation. "
                               "Should be 'linear' or 'circular'." },

        'mem_budget': {
            'type': 'integer',
            'min': 1024,
            'default': 1024*1024*1024,
            '__description__': "Memory budget for solving a single "
                               "tile of the problem on a CPU/GPU "
                               "in bytes." },

        'source_batch_size': {
            'type': 'integer',
            'min': 0,
            'default': 500 },

        'version': {
            'type': 'string',
            'default': 'tf' },
    }

    return DescriptionValidator(schema)
