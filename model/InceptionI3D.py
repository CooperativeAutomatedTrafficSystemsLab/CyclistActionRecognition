import tensorflow as tf


## Tensorflow implementation of Inception I3D architecture from https://github.com/deepmind/kinetics-i3d
#
class InceptionI3D:

    ## Constructor
    #
    # @param: x (tf graph/placeholder): input image sequence
    # @param: training (tf.placeholder(bool)): whether or not training is active
    # @param: activation (tf activation): activation funtion used between layers
    # @param: use_batch_norm (bool): whether or not batch normalizatino should be used (currently not implemented)
    def __init__(self, x, training, activation=tf.nn.selu, use_batch_norm=True):
        self.x = x
        self.training = training
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.model = self._build(inputs=x, activation=activation)

    ## Build graph
    #
    # @param: inputs (tf graph/placeholder): input image sequence
    # @param: activation (tf activation): activation function used between layers
    # @retval: built graph
    def _build(self, inputs, activation=tf.nn.selu):
        net = tf.expand_dims(inputs, -1)
        name = 'Conv3d_1a_7x7'
        net = tf.layers.conv3d(net,
                               filters=64,
                               kernel_size=7,
                               strides=[2, 2, 2],
                               padding='same',
                               name=name,
                               activation=activation)

        name = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net,
                               ksize=[1, 3, 3, 1, 1],
                               strides=[1, 2, 2, 1, 1],
                               padding='SAME',
                               name=name)
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self.training)
        name = 'Conv3d_2b_1x1'
        net = tf.layers.conv3d(net,
                               filters=64,
                               kernel_size=[1, 1, 1],
                               padding='same',
                               name=name,
                               activation=activation)
        name = 'Conv3d_2c_3x3'
        net = tf.layers.conv3d(net, filters=192, kernel_size=[3, 3, 3],
                               name=name,
                               padding='same',
                               activation=activation)
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 1, 1], strides=[1, 2, 2, 1, 1],
                               padding='SAME', name=name)
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self.training)
        name = 'Mixed_3b'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=96, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=128, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=16, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=32, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=32, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_3c'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=192, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=32, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=96, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        name = 'MaxPool3d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=name)
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_4b'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=192, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=96, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=208, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=16, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=48, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_4c'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=160, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=112, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=224, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=24, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=64, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_4d'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=256, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=24, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=64, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_4e'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=112, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=144, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=288, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=32, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=64, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=64, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_4f'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=256, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=160, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=320, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=32, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=128, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        name = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=name)
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self.training)
        name = 'Mixed_5b'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=256, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=160, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=320, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=32, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=128, kernel_size=[3, 3, 3],
                                            name='Conv3d_0a_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        name = 'Mixed_5c'
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv3d(net, filters=384, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv3d(net, filters=192, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_1 = tf.layers.conv3d(branch_1, filters=384, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv3d(net, filters=48, kernel_size=[1, 1, 1],
                                            name='Conv3d_0a_1x1',
                                            padding='same',
                                            activation=activation)
                branch_2 = tf.layers.conv3d(branch_2, filters=128, kernel_size=[3, 3, 3],
                                            name='Conv3d_0b_3x3',
                                            padding='same',
                                            activation=activation)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding='SAME',
                                            name='MaxPool3d_0a_3x3')
                branch_3 = tf.layers.conv3d(branch_3, filters=128, kernel_size=[1, 1, 1],
                                            name='Conv3d_0b_1x1',
                                            padding='same',
                                            activation=activation)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            if self.use_batch_norm:
                net = tf.layers.batch_normalization(net, training=self.training)

        return net
