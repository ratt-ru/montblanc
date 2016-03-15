from attrdict import AttrDict
import numpy as np

from montblanc.src_types import SOURCE_VAR_TYPES
from montblanc.enums import DIMDATA

DEFAULT_DESCRIPTION = 'An inexplicable dimension!'

def create_dim_data(name, dim_data, **kwargs):
    """
    Create a dimension data dictionary from dim_data
    and keyword arguments. Keyword arguments will be
    used to update the dictionary.

    Arguments
    ---------
        name : str
            Name of the dimension
        dim_data : integer or another dimension data dictionary
            If integer a fresh dictionary will be created, otherwise
            dim_data will be copied.

    Returns
    -------
        A dimension data dictionary

    """

    # If dim_data is an integer, start constructing a dictionary from it
    if isinstance(dim_data, (int, long, np.integer)):
        dim_data = { DIMDATA.NAME : name, DIMDATA.GLOBAL_SIZE : dim_data }
    elif not isinstance(dim_data, dict):
        raise TypeError('dim_data must be an integer or a dict')

    if not isinstance(dim_data, AttrDict):
        dim_data = AttrDict(dim_data.copy())

    # Now update the dimension data from any keyword arguments
    dim_data.update(kwargs)

    # Need a name and global_size at minimum
    for v in (DIMDATA.NAME, DIMDATA.GLOBAL_SIZE):
        assert v in dim_data, ("Dictionary for dimension '{d}' "
            "must have a '{e}' entry").format(d=name, e=v)

    global_size = dim_data[DIMDATA.GLOBAL_SIZE]

    # Configure local size if not present
    if DIMDATA.LOCAL_SIZE not in dim_data:
        dim_data[DIMDATA.LOCAL_SIZE] = global_size

    # Configure extents if not present
    if DIMDATA.EXTENTS not in dim_data:
        dim_data[DIMDATA.EXTENTS] = [0, global_size]

    # Configure dimension if not present
    if DIMDATA.DESCRIPTION not in dim_data:
        dim_data[DIMDATA.DESCRIPTION] = DEFAULT_DESCRIPTION

    # Turn safety on by default
    if DIMDATA.SAFETY not in dim_data:
        dim_data[DIMDATA.SAFETY] = True

    # Don't allow zero sized dimensions by default
    if DIMDATA.ZERO_VALID not in dim_data:
        dim_data[DIMDATA.ZERO_VALID] = False

    return dim_data

def update_dim_data(dim, update_dict):
    """
    Sanitised dimension data update

    Arguments
    ---------
        dim : dict
            dimension data dictionary
        update_dict : dict
            Dictionary containing a list of key-values
            for updating dim
    """
    import collections

    name = dim[DIMDATA.NAME]

    # Update options if present
    if DIMDATA.SAFETY in update_dict:
        dim[DIMDATA.SAFETY] = update_dict[DIMDATA.SAFETY]

    if DIMDATA.ZERO_VALID in update_dict:
        dim[DIMDATA.ZERO_VALID] = update_dict[DIMDATA.ZERO_VALID]

    if DIMDATA.LOCAL_SIZE in update_dict:
        if dim[DIMDATA.SAFETY] is True:
            raise ValueError(("Modifying local size of dimension '{d}' "
                "is not allowed by default. If you are sure you want "
                "to do this add a '{s}' : 'False' entry to the "
                "update dictionary.").format(d=name, s=DIMDATA.SAFETY))

        if dim[DIMDATA.ZERO_VALID] is False and update_dict[DIMDATA.LOCAL_SIZE] == 0:
            raise ValueError(("Modifying local size of dimension '{d}' "
                "to zero is not valid. If you are sure you want "
                "to do this add a '{s}' : 'True' entry to the "
                "update dictionary.").format(d=name, s=DIMDATA.ZERO_VALID))

        dim[DIMDATA.LOCAL_SIZE] = update_dict[DIMDATA.LOCAL_SIZE]

    if DIMDATA.EXTENTS in update_dict:
        exts = update_dict[DIMDATA.EXTENTS]
        if (not isinstance(exts, collections.Sequence) or len(exts) != 2):
            raise TypeError("'{e}' entry in update dictionary "
                "must be a sequence of length 2.".format(e=DIMDATA.EXTENTS))

        dim[DIMDATA.EXTENTS] = [v for v in exts[0:2]]

    # Check that we've been given valid values
    check_dim_data(dim)

def check_dim_data(dim_data):
    """ Sanity check the contents of a dimension data dictionary """
    ls, gs, E, name, zeros = (dim_data[DIMDATA.LOCAL_SIZE],
        dim_data[DIMDATA.GLOBAL_SIZE],
        dim_data[DIMDATA.EXTENTS],
        dim_data[DIMDATA.NAME],
        dim_data[DIMDATA.ZERO_VALID])

    # Sanity check dimensions
    assert 0 <= ls <= gs, \
        ("Dimension '{n}' local size {l} is greater than "
        "it's global size {g}").format(
            n=name, l=ls, g=gs)

    assert E[1] - E[0] <= ls, \
        ("Dimension '{n}' local size {l} is greater than "
        "it's extents [{e0}, {e1}]").format(
            n=name, l=ls, e0=E[0], e1=Ep[1])

    if zeros:
        assert 0 <= E[0] <= E[1] <= gs, (
            "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                .format(d=name, gs=gs, e0=E[0], e1=E[1]))
    else:
        assert 0 <= E[0] < E[1] <= gs, (
            "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                .format(d=name, gs=gs, e0=E[0], e1=E[1]))    
