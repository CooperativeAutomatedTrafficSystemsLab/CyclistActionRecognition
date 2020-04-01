from functools import partial, reduce
from itertools import product
from collections import Mapping
import numpy as np
import operator
import six


## Grid of parameters with a discrete number of values for each.
# Can be used to iterate over parameter value combinations with the
# Python built-in function iter.
# Copied from scikit-learn
class ParameterGrid(object):

    ## Constructor
    #   @param param_grid:
    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    ## Iterate over the points in the grid.
    #   @return iterator over dict of string to any
    #           Yields dictionaries mapping each estimator parameter to one of its
    #           allowed values.
    def __iter__(self):

        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    ## Number of points on the grid
    #   @return: number of points in the grid
    def __len__(self):

        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    ## Get the parameters that would be ``ind``th in iteration
    #   @param ind: The iteration index
    #   @return:    dict of string to any
    #               Equal to list(self)[ind]
    def __getitem__(self, ind):

        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memorize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


## checks a parameter grid for validity
#
# @param param_grid:
def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, six.string_types)):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))
