## @package layers
#  A collection of baysic tensorflow layers

import tensorflow as tf

## Dropout layer, applyed before a nonlinear layer
#  @param X (tensor)                layer input.
#  @param prob (float)              keep probability of a neuron
#  @param train (tensor, bool)      whether to return the output in training mode or in inference mode.
#  @param name (str)                the operations name
#  @return (tensor)                 layer with dropped out neurons
def Dropout(X, prob=0.7, train=tf.constant(False), name='Dropout'):
    from tensorflow.contrib.distributions import Bernoulli
    if not isinstance(prob, float) or prob > 1.0 or prob < 0.0:
        raise ValueError('Encountered illegal value for param (prob), expecting float between 0 and 1')
    with tf.name_scope(name):
        Dropout_Mask = tf.diag(Bernoulli(probs=prob, dtype=tf.float32).sample((tf.shape(X)[-1],)), 'Dropout_Mask')
        X_dropped = tf.matmul(X, Dropout_Mask)
    return tf.cond(tf.equal(train, tf.constant(True)), lambda: X_dropped, lambda: X)


## A Fully Conected Layer.
#   Example call for a FCL with 3 inputs and 2 outputs, and relu nonliterary:
#
#   my_fcl_layer = Dense(3,2)
#   output = my_fcl_layer(input, activation = tf.nn.relu)
#   reg = my_fcl_layer.regularization(0.7, type='LASSO')
class Dense:
    ## constructor, initilizes the fully connected layer
    #  @param n_in (int)    number input neurons.
    #  @param n_out (int)   number output neurons.
    #  @param xavier_initializer (bool) whether to use xavier initialization
    #  @raises (ValueError) if n_in is not int or less than 0
    #  @raises (ValueError) if n_out is not int or less than 0
    #  @raises (ValueError) if xavier is not bool
    def __init__(self, n_in, n_out, xavier_initializer=False):

        if not (isinstance(n_in, int) and n_in > 0):
            raise ValueError('Encountered illegal value for param (n_in), expecting integer greater 0')
        if not (isinstance(n_out, int) and n_out > 0):
            raise ValueError('Encountered illegal value for param (n_out), expecting integer greater 0')
        if not isinstance(xavier_initializer, bool):
            raise ValueError('Encountered illegal value for param (xavier_initializer), bool')

        if xavier_initializer:
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.W = tf.Variable(initializer([n_in, n_out]), name='weights')  # Weights
            self.b = tf.Variable(tf.ones([n_out]), name='bias')  # Bias
        else:
            self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')  # Weights
            self.b = tf.Variable(tf.truncated_normal([n_out], stddev=0.01), name='bias')  # Bias

    ## call, perfoms a foreward pass through the layer
    # The backward pass is computed by tensorflow and therefore does not have to be defined!
    #  @param X (tensor)    layers input
    #  @param activation (None/lambda)  non linear activation function (use tf functions!)
    #  @returns  (tensor)   layers output
    def __call__(self, X, activation=None):
        if activation == None:
            # linear layer
            output = tf.matmul(X, self.W) + self.b
        else:
            # non linear layer
            output = activation(tf.matmul(X, self.W) + self.b)
        if self.W.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    ## regularization of the layer (l1 or l2)
    #  In order to use it with the gradient descent,
    #  make sure to sum up the regularization terms of all of all layers and add it to the loss function!
    #  @param lam (float)   momentum of the regularization
    #  @param type (None/str) string to determine the type of the regularization
    #  @return  (tensor)    regularization term
    def regularization(self, lam, type=None):
        if type == 'l1' or type == 'LASSO':
            with tf.name_scope(type + '_Regularization'):
                W_reg = l1_regularization(self.W, lam)
                b_reg = l1_regularization(self.b, lam)
                return W_reg + b_reg
        elif type == 'l2' or type == 'Ridge' or type == 'Tikhonov':
            with tf.name_scope(type + '_Regularization'):
                W_reg = l2_regularization(self.W, lam)
                b_reg = l2_regularization(self.b, lam)
                return W_reg + b_reg
        else:
            return tf.constant(0)


## L1 Regularization
#  @param X (tensor)    input.
#  @param lam (float)   momentum of the regularization
#  @return  (tensor)    regularization
#  @raises (ValueError) if lam is not float or not within the interval [0,1]
def l1_regularization(X, lam):
    if not isinstance(lam, float) or lam > 1.0 or lam < 0.0:
        raise ValueError('Encountered illegal value for param (lam), expecting float between 0 and 1')

    return lam / 2.0 * tf.reduce_sum(tf.square(X))


## L2 Regularization
#  @param X (tensor)    input.
#  @param lam (float)   momentum of the regularization
#  @return  (tensor)    regularization term
#  @raises (ValueError) if lam is not float or not within the interval [0,1]
def l2_regularization(X, lam):
    if not isinstance(lam, float) or lam > 1.0 or lam < 0.0:
        raise ValueError('Encountered illegal value for param (lam), expecting float between 0 and 1')

    return lam * tf.reduce_sum(tf.abs(X))


## MultiLSTMCell
#  @param X (tensor)                            layers input.
#  @param state_size_layer (list of int)        number of neurons for each layer.
#  @param activation_LSTM_cell (None/lambda)    activation function of the inner states of LSTM (use tf functions!).
#  @param dynamic (bool)                        use dynamic unrolling through time.
#  @param use_peepholes (bool)                  use peephole connections.
#  @param dropout_keep_prob (float)             keeping percentage after dropout
#  @param wrapped (bool)                        wrap LSTM with a fully connected layer
#  @param activation_wrapper (None/lambda)      activation function of the fully connected layer
#  @param output_size (int)                     output_size of the fully connected layer
#  @param sequence_length (list of int)         length of the input sequence (vector sized [batch_size])
#  @return output (tensor)                      output
#  @return current_state (tensor)               the final state
def multi_LSTM_cell(X, state_size_layer=[1], activation_LSTM_cell=tf.nn.relu, dynamic=True, use_peepholes=False,
                    dropout_keep_prob=1.0, wrapped=False, activation_wrapper=None, output_size=1, sequence_length=None):
    n_layer = len(state_size_layer)
    cell = [None] * n_layer
    for layer in range(0, n_layer):
        with tf.name_scope("LSTMCell_" + str(layer + 1)):
            cell[layer] = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(state_size_layer[layer], use_peepholes=use_peepholes,
                                        state_is_tuple=True, activation=activation_LSTM_cell),
                output_keep_prob=dropout_keep_prob)

    if wrapped:
        cell[-1] = tf.contrib.rnn.OutputProjectionWrapper(cell[-1], output_size=output_size,
                                                          activation=activation_wrapper)

    # Stack the LSTM Cells
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)

    if dynamic is False:
        states_series, current_state = tf.nn.static_rnn(stacked_cells, X, sequence_length=sequence_length,
                                                        dtype=tf.float32)
    else:
        states_series, current_state = tf.nn.dynamic_rnn(stacked_cells, X, sequence_length=sequence_length,
                                                         dtype=tf.float32)

    return states_series, current_state


## MultiGRUCell
#  @param X (tensor)                            layers input.
#  @param state_size_layer (list of int)        number of neurons for each layer.
#  @param activation_GRU_cell (None/lambda)     activation function of the inner states of GRU (use tf functions!).
#  @param dynamic (bool)                        use dynamic unrolling through time.
#  @param dropout_keep_prob (float)             keeping percentage after dropout
#  @param wrapped (bool)                        wrap LSTM with a fully connected layer
#  @param activation_wrapper (None/lambda)      activation function of the fully connected layer
#  @param output_size (int)                     output_size of the fully connected layer
#  @param sequence_length (list of int)         length of the input sequence (vector sized [batch_size])
#  @return output (tensor)                      output
#  @return current_state (tensor)               the final state
def multi_GRU_cell(X, state_size_layer=[1], activation_GRU_cell=tf.nn.relu, dynamic=True, dropout_keep_prob=1.0,
                   wrapped=False, activation_wrapper=None, output_size=1, sequence_length=None):
    n_layer = len(state_size_layer)
    cell = [None] * n_layer
    for layer in range(0, n_layer):
        with tf.name_scope("GRUCell_" + str(layer + 1)):
            cell[layer] = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(state_size_layer[layer], activation=activation_GRU_cell),
                output_keep_prob=dropout_keep_prob)

    if wrapped:
        cell[-1] = tf.contrib.rnn.OutputProjectionWrapper(cell[-1], output_size=output_size,
                                                          activation=activation_wrapper)

    # Stack the GRU Cells
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)

    if dynamic is False:
        states_series, current_state = tf.nn.static_rnn(stacked_cells, X, sequence_length=sequence_length,
                                                        dtype=tf.float32)
    else:
        states_series, current_state = tf.nn.dynamic_rnn(stacked_cells, X, sequence_length=sequence_length,
                                                         dtype=tf.float32)

    return states_series, current_state


## MultiFCLCell
#  @param X (tensor)                            layers input.
#  @param state_size_layer (list of int)        number of neurons for each layer.
#  @param activation (None/lambda)              activation function of the inner states of GRU (use tf functions!).
#  @param dropout_keep_prob (float)             keeping percentage after dropout
def multi_FCL_cell(X, state_size_layer=[1], activation=tf.nn.relu, dropout_keep_prob=1.0):
    # Flatten feature vector for FCL usage
    next_layer_in = tf.layers.Flatten()(X)

    # Loop over all hidden Layers
    for n_neurons in state_size_layer:
        dropout = tf.layers.dropout(next_layer_in, rate=dropout_keep_prob)
        next_layer_in = tf.layers.dense(dropout, units=n_neurons, activation=activation)

    return next_layer_in


## Hidden_Architecture
#  high level method for defining a hidden architecture
#  @param X (tensor)            layers input.
#  @param architecture (dict)   hidden architecture for the network
#                               # example: {'FCL': {'activation': tf.nn.relu, 'layers': [100,100], 'keep_prop': 1.0}}
#                               # -> one fully connected layer with two hidden layers a 100 neurons, relu non-linearity
#                               #    and a dropout keep prob of 100 percent
def hidden_architecture(X, architecture={'FCL': {'activation': tf.nn.relu, 'layers': [100], 'keep_prop': 1.0}}):
    next_in = X
    # Hidden Layer
    for cell_type, cells in architecture.items():
        if cell_type == 'LSTM':
            with tf.name_scope('LSTM_Cells'):
                next_in, _ = multi_LSTM_cell(next_in, cells['layers'],
                                             activation_LSTM_cell=cells['activation'],
                                             dropout_keep_prob=cells['keep_prop'])

        elif cell_type == 'GRU':
            with tf.name_scope('GRU_Cells'):
                next_in, _ = multi_GRU_cell(next_in, cells['layers'],
                                            activation_GRU_cell=cells['activation'],
                                            dropout_keep_prob=cells['keep_prop'])

        elif cell_type == 'FCL':
            with tf.name_scope('FCL_Cells'):
                next_in = multi_FCL_cell(next_in, cells['layers'],
                                         activation=cells['activation'],
                                         dropout_keep_prob=cells['keep_prop'])
        else:
            print(Warning('WARNING: unkown cell type ' + cell_type + ' using FCL instead!'))
    y = next_in
    return y
