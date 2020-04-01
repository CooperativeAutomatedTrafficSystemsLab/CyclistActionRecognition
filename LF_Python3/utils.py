import warnings
import os
import numpy as np


def remove_addresses(string):
    import re
    found = re.findall(r'0x[0-9A-F]+', string, re.I)
    for x in found:
        string = string.replace(x, '')
    return string


def safe_indexing(X, indices):
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            warnings.warn("Copying input dataframe for slicing.")
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def as_memmap(X, name):
    from tempfile import mkdtemp
    tmp_dir = mkdtemp()

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filename = os.path.join(tmp_dir, name + '.mmap')
    shape = X.shape

    memmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)
    memmap[:] = X[:]
    memmap.flush()
    return memmap, filename, shape


def clean_up_memmap(mmap):
    try:
        os.remove(mmap)
    except:
        print('Could not clean-up automatically.')


def read_memmap(path, shape):
    return np.memmap(path, dtype=np.float32, mode='r', shape=shape)


def batched_mean_map(X_path, X_shape, name=None):
    X_map = read_memmap(X_path, X_shape)
    sum = 0
    n_batches = X_shape[0]
    for batch in range(n_batches):
        sum += np.mean(X_map[batch, :])
    mean = sum / n_batches
    return mean


def batched_mean(X, name=None):
    sum = 0
    n_batches = X.shape[0]
    for batch in range(n_batches):
        sum += np.mean(X[batch, :])
    mean = sum / n_batches
    return mean


def detect_overridden(cls, obj):
    common = set(cls.__dict__.keys()) & set(obj.__class__.__dict__.keys())
    diff = [m for m in common if cls.__dict__[m] != obj.__class__.__dict__[m]]
    return diff
