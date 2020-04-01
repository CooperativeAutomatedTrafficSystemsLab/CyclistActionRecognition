import numpy as np
from sklearn.utils import shuffle


## Array object used used for shared memory applications.
#  Can return either references or copies of the original array by indexing or slicing.
class array:
    ##  constructor of the dtype kav_array
    #   @param values (list): list of values which are added to the shm array
    #   @param dtype (np.dtype): type of created array
    #   @param normalization_factor (float): values are multiplied by normalization_factor when added to array
    def __init__(self, values, dtype=np.float32, normalization_factor=1.0):
        # Expand dims by one for the indexing
        self.dtype = dtype
        self.references = [np.array([i]).astype(self.dtype) * normalization_factor for i in values]
        self.normalization_factor = normalization_factor

    ## returns references at certain indices from self.references
    # @param indices (list): list of indices
    # @return list containing reference of elements in self.values at indices
    def __getitem__(self, indices):
        temp = array([], dtype=self.dtype)
        if isinstance(indices, slice):
            temp.references = self.references[indices]
        else:
            temp.references = [self.references[i] for i in indices]
        return temp

    ## sets item value at certain indices
    # @param value: value to set
    # @param index (int): index where name should be changed
    def __setitem__(self, index, value):
        self.references[index][0] = value

    ## return length of references
    def __len__(self):
        return len(self.references)

    ## returns copy of values at certain indices from self.references
    # @param indices (list): list of indices
    # @return list containing values of elements in self.values at indices
    def get_copy(self, indices=None):
        if indices is None:
            return np.array(self.references).squeeze(axis=1)
        if isinstance(indices, slice):
            if indices.start == len(self.references):
                return []
            else:
                return np.array(self.references[indices]).squeeze(axis=1)
        else:
            return np.array([self.references[i][0] for i in indices])

    ## extends self.references by values
    # @param values (list): list which extends self.references
    def extend(self, values):
        temp_list = [np.array([i]).astype(self.dtype) * self.normalization_factor for i in values]
        self.references.extend(temp_list)

    ## appends value to self.references
    # @param value: value which is appended to self.references
    def append(self, value):
        self.references.append(np.array([value]).astype(self.dtype) * self.normalization_factor)

    ## return shape of original array
    def shape(self):
        data_shape = np.shape(self.references[0])[1:]
        return (len(self),) + data_shape


## Shuffle array objects in a consistent way
# @param *arrays: sequence of array objects
# @param random_state: int, RandomState instance or None, optional (default=None)
# @return sequence of array objects with shuffled references (no copy). The original array objects are not influenced.
def shuffle_arrays(*arrays, **options):
    shuffled_arrays = [array([], dtype=a.dtype) for a in arrays]
    references = [a.references for a in arrays]
    shuffled_references = shuffle(*tuple(references), **options)
    for sh_array, reference in zip(shuffled_arrays, shuffled_references):
        sh_array.references = reference
    return shuffled_arrays
