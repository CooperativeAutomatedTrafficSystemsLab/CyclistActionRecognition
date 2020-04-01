import numpy as np


class Homogeneous:
    ## converts a tensor to homogeneous coordinates
    #
    @staticmethod
    def to_homogeneous_coordinates(a):
        return np.concatenate((a, np.ones_like(a[..., 0:1])), axis=-1)

    ## converts a tensor back from homogeneous to normal
    #
    @staticmethod
    def to_normal_coordinates(a):
        return a[..., 0:-1]

    ## builds the euclidean norm over a tensors last axis
    #
    # @param tensor: (ndarray)
    # @return (ndarray) tensors norm over the last axis
    @staticmethod
    def norm(tensor):
        return np.expand_dims(np.einsum('...a,...a->...', tensor, tensor, optimize=True), axis=-1)

    ## applies a transformation matrix or a collection of transformation matrices on the last axis of the tensor
    #
    # @param a: (ndarray)  tensor to be transformed
    # @param M: (ndarray)   the transformation matrix or a collection of transformation matrices (stacked in a ndarray)
    # @param swap (bool)    option for swapping axis 0 and 1
    # @return (ndarray)     transformed array
    @staticmethod
    def transform_array_by_matrix(a, M):
        # does Ma
        return np.einsum('...ij,...j->...i', M, a, optimize=True)

    ## applies a transformation matrix or a collection of transformation matrices on the last two axis of the tensor
    # Does O=MAM^T
    # @param A: (ndarray)  tensor to be transformed
    # @param M: (ndarray)   the transformation matrix or a collection of transformation matrices (stacked in a ndarray)
    # @return (ndarray)     transformed matrix/matrices
    @staticmethod
    def transform_matrix_by_matrix(A, M):
        # Transpose matrix
        M_transposed = np.einsum('...ab->...ba', M, optimize=True)
        # Do B=AM^T
        B = np.einsum('...ab,...bc->...ac', A, M_transposed, optimize=True)
        # Do MA
        O = np.einsum('...ab,...bc->...ac', M, B, optimize=True)
        return O

    ## builds a collection of scale matrices
    #
    # @param shape: (tuple) shape of the matrix to be created
    # @param scale (float)  the factor to be scaled with
    # @return: (ndarray) collection of scale matrices
    @staticmethod
    def build_scale_matrices(scale, shape=(3, 3)):
        # get identity matrices
        scale_matrix = Homogeneous.build_diagonal_matrix_like(shape)
        for idx in range(scale_matrix.shape[-1] - 1):
            # fill scale values
            scale_matrix[..., idx, idx] = scale
        return scale_matrix

    ## builds a collection of translation matrices
    #
    # @param shape: (tuple) shape of the matrix to be created
    # @param translation_vector (ndarray)  the translation vectors
    # @return: (ndarray) collection of translation matrices
    @staticmethod
    def build_translation_matrices(translation_vector, shape=(3, 3)):
        # get identity matrices
        translation_matrix = Homogeneous.build_diagonal_matrix_like(shape=shape)
        for idx in range(translation_matrix.shape[-1]):
            # fill translation values
            translation_matrix[..., idx, translation_matrix.shape[-1] - 1] = -translation_vector[..., idx]
        return translation_matrix

    ## builds a collection of roatation matrices
    #
    # @param shape: (tuple) shape of the matrix to be created
    # @param rotation_angle: (ndarray)  the translation angles
    # @return: (ndarray) collection of roatation matrices
    @staticmethod
    def build_rotation_matrices(rotation_angle, shape=(3, 3)):
        # get identity matrices
        rotation_matrix = Homogeneous.build_diagonal_matrix_like(shape)
        rotation_matrix[..., 0, 0] = np.cos(rotation_angle)
        rotation_matrix[..., 0, 1] = np.sin(rotation_angle)
        rotation_matrix[..., 1, 0] = -np.sin(rotation_angle)
        rotation_matrix[..., 1, 1] = np.cos(rotation_angle)
        return rotation_matrix

    ## creates a collection of diagonal matrices to have complied with the input tensor X
    #
    # @param shape: (tuple) shape of the matrix to be created
    # @return: (ndarray) collection of diagonal matrices
    @staticmethod
    def build_diagonal_matrix_like(shape, dtype=np.float32):
        diagonal_matrix = np.zeros(shape=shape, dtype=dtype)
        # fill diagonal
        for idx in range(diagonal_matrix.shape[-1]):
            diagonal_matrix[..., idx, idx] = 1
        return diagonal_matrix

    ## builds the determinant over the last two axes of a tensor
    @staticmethod
    def det(M):
        return np.linalg.det(M)

    ## builds the inverse over the last two axes of a tensor
    @staticmethod
    def inv(M, e_min=1e-5):
        # if all determinants are over the stability constraint, build the inverse
        if np.all(Homogeneous.det(M) > e_min):
            return np.linalg.inv(M)
        # else build the pseudo inverse
        else:
            return np.linalg.pinv(M)

    ## builds the quadratic form of matrix M and arrays a and b
    #  uses last axes of a and b and the last two axes of M
    @staticmethod
    def quadratic_form(a, M, b):
        return np.einsum('...a,...ab,...b->', a, M, b, optimize=True)

    ## builds the Mahalanobis distance form with matrix M and array a
    #  uses last axis of a and the last two axes of M
    @staticmethod
    def Mahalanobis_distance(a, M):
        return Homogeneous.quadratic_form(a, M, a)
