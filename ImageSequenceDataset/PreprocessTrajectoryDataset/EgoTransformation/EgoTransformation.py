import numpy as np
from .Homogeneous import Homogeneous


## Ego Transformation
#  Transformation of a trajectory from world-coordinates into ego-coordinates and vice versa
class Ego:
    ## constructor
    #
    # @param: scale (int)   the scale between ego and world coordinates
    def __init__(self, scale=1.0):
        self.scale = scale

    ## Transforms a trajectory array into ego coordinates
    #
    # @param X: (ndarray)            input trajectory
    # @param y: (optional ndarray)   ground truth trajectory
    # @return:  (ndarray)            input trajectory in ego coordinates and if passed a ground truth trajectory in ego coordinates
    def transform(self, X, y=None, orientation=None):
        # Transform trajectory to homogeneous coordinates
        X_homogen = Homogeneous.to_homogeneous_coordinates(X)

        # get desired shape of the matrix collection
        shape = self.get_matrix_shape(X_homogen)

        # Translate such that the last known position is at the coordinate systems origin
        # get the last known position
        self.reference_position = X_homogen[..., -1, :]

        # construct translation matrix (homogeneous coordinates)
        translation_matrix = Homogeneous.build_translation_matrices(self.reference_position, shape=shape)

        # swap axes
        X_homogen = np.swapaxes(X_homogen, 0, 1)

        # apply transformation
        X_translated = Homogeneous.transform_array_by_matrix(X_homogen, translation_matrix)

        # if no orientation was passed, calculate it from passed trajectory
        if orientation is None:
            # Rotate such that orientation of the past positions is on the y-axis of the coorinate system
            # get orientation as sum of normalized location vectors
            X_normed = X_translated / Homogeneous.norm(X_translated)
            direction = np.sum(X_normed, axis=0)
            # determine angle between the orientation and the y-axis
            self.rotation_angle = np.arctan2(direction[..., 0], -direction[..., 1])
        else:
            self.rotation_angle = orientation
        # construct rotation matrix (homogeneous coordinates)
        rotation_matrix = Homogeneous.build_rotation_matrices(self.rotation_angle, shape=shape)
        # apply transformation
        X_rotated = Homogeneous.transform_array_by_matrix(X_translated, rotation_matrix)

        # scale coordinates by specified scale
        # construct scale matrix
        scale_matrix = Homogeneous.build_scale_matrices(self.scale, shape=shape)
        # apply transformation
        X_scaled = Homogeneous.transform_array_by_matrix(X_rotated, scale_matrix)

        # convert from homogeneous to normal coordinates
        X_ego = Homogeneous.to_normal_coordinates(X_scaled)

        # swap axes back
        X_ego = np.swapaxes(X_ego, 0, 1)

        # if a ground truth trajectory is passed, transform the ground truth trajectory
        if not isinstance(y, type(None)):
            # check if y has same dimension as X
            if X.shape[2] != y.shape[2]:
                # expand third dimension to X dimension and fill with zeros
                y_same_dim = np.zeros((y.shape[0], y.shape[1], X.shape[2]))
                y_same_dim[:, :, :y.shape[2]] = y
            else:
                y_same_dim = y

            y_homogen = Homogeneous.to_homogeneous_coordinates(y_same_dim)
            y_homogen = np.swapaxes(y_homogen, 0, 1)
            y_translated = Homogeneous.transform_array_by_matrix(y_homogen, translation_matrix)
            y_rotated = Homogeneous.transform_array_by_matrix(y_translated, rotation_matrix)
            y_scaled = Homogeneous.transform_array_by_matrix(y_rotated, scale_matrix)
            y_ego = Homogeneous.to_normal_coordinates(y_scaled)
            # discard additional axes
            if X.shape[2] != y.shape[2]:
                y_ego = y_ego[:, :, :y.shape[2]]
            y_ego = np.swapaxes(y_ego, 0, 1)
            return X_ego, y_ego
        return X_ego

    ## Transforms a trajectory array from ego to world coordinates
    #
    # @param X: (ndarray)            input trajectory
    # @param y: (optional ndarray)   ground truth trajectory
    # @return:  (ndarray)            input trajectory in ego coordinates and if passed a ground truth trajectory in world coordinates
    def inverse_transform(self, X, y=None):

        if not hasattr(self, 'scale') or not hasattr(self, 'rotation_angle') or not hasattr(self, 'reference_position'):
            raise AttributeError('Transformation parameter are not set')

        # Transform trajectory to homogeneous coordinates
        X_homogen = Homogeneous.to_homogeneous_coordinates(X)
        shape = self.get_matrix_shape(X_homogen)

        # swap axes
        X_homogen = np.swapaxes(X_homogen, 0, 1)

        # scale coordinates back
        # construct inverse scale matrix
        inv_scale_matrix = Homogeneous.build_scale_matrices(1 / self.scale, shape=shape)
        # apply transformation
        X_inv_scaled = Homogeneous.transform_array_by_matrix(X_homogen, inv_scale_matrix)

        # rotate back to world coordinates
        inv_rotation_matrix = Homogeneous.build_rotation_matrices(-self.rotation_angle, shape=shape)
        # construct inverse rotation matrix (homogeneous coordinates)
        X_inv_rotated = Homogeneous.transform_array_by_matrix(X_inv_scaled, inv_rotation_matrix)

        # translate back to world coordinates
        inv_translation_matrix = Homogeneous.build_translation_matrices(-self.reference_position, shape=shape)
        # construct inverse translation matrix (homogeneous coordinates)
        X_inv_translated = Homogeneous.transform_array_by_matrix(X_inv_rotated, inv_translation_matrix)

        # convert from homogeneous to normal coordinates
        X_world = Homogeneous.to_normal_coordinates(X_inv_translated)

        # swap axes back
        X_world = np.swapaxes(X_world, 0, 1)

        # if a ground truth trajectory is passed, transform the ground truth trajectory
        if not isinstance(y, type(None)):
            y_homogen = Homogeneous.to_homogeneous_coordinates(y)
            y_homogen = np.swapaxes(y_homogen, 0, 1)
            y_inv_scaled = Homogeneous.transform_array_by_matrix(y_homogen, inv_scale_matrix)
            y_inv_rotated = Homogeneous.transform_array_by_matrix(y_inv_scaled, inv_rotation_matrix)
            y_inv_translated = Homogeneous.transform_array_by_matrix(y_inv_rotated, inv_translation_matrix)
            y_world = Homogeneous.to_normal_coordinates(y_inv_translated)
            y_world = np.swapaxes(y_world, 0, 1)
            return X_world, y_world

        return X_world

    def get_matrix_shape(self, X):
        matrix_shape = list(X.shape[:-2])
        matrix_shape.append(X.shape[-1])
        matrix_shape.append(X.shape[-1])
        return matrix_shape

    def get_transform_parameters(self):
        return self.reference_position, self.rotation_angle, self.scale

    def set_transform_matrices(self, reference_position, rotation_angle, scale):
        self.reference_position = reference_position
        self.rotation_angle = rotation_angle
        self.scale = scale


## Crops a Sequence into Samples by appling a rolling window
#
# @param a:     the trajectory
# @param n_in:  size of the input window
# @param n_out: size of the ground truth window
# @param dims:  the dimensions of the trajectory to be used
# @return:  the cropped input trajectory and if n_out is not none the cropped
# @raises:  Value errors
def crop_trajectory(a, n_in, n_out=None, dims=2):
    if dims > a.shape[-1]:
        raise ValueError('The trajectory only has %d dimensions' % a.shape[-1])
    if not isinstance(n_out, type(None)):
        X = rolling_window(a[..., :-n_out, :], shape=(n_in, dims))
        y = rolling_window(a[..., n_in:, :], shape=(n_out, dims))
        return X, y
    else:
        X = rolling_window(a[..., :, :], shape=(n_in, dims))
        return X


## Crops orientations to in and output sizes
#
# @param orientation (list): List of orientations
# @param n_in (int):  size of the input window
# @param n_out (int): size of the ground truth window
def crop_orientation(orientation, n_in, n_out=None):
    if n_out is None:
        return orientation[n_in - 1:]
    else:
        return orientation[n_in - 1:-n_out]


## sliding window over a trajectory
#
# @param a: (ndarray)   input trajectory (e.g. 2d trajectory got the shape (n_samples,2))
# @param shape: (tuple) shape for the sliding window (e.g. for 50 samples of a 2d trajectory use (50,2))
# @return: (ndarray)    rolling window (e.g. for 50 samples of a 2d trajectory, the result has the shape (n_samples,50,2))
def rolling_window(a, shape):
    s = (a.shape[-2] - shape[-2] + 1,) + (a.shape[-1] - shape[-1] + 1,) + shape
    strides = a.strides + a.strides
    astrided_array = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze()
    return astrided_array
