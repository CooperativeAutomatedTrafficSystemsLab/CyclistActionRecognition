import numpy as np
import tensorflow as tf
import os
import pickle
import time
import json
import inspect
import hashlib
from .tf_memory import sort_gpus, mask_gpus
from .utils import remove_addresses, detect_overridden
from .helper import shared_memory as shm
from filelock import FileLock

try:
    __cuda_devices__ = os.environ['CUDA_VISIBLE_DEVICES']
except:
    __cuda_devices__ = None


class BasicNetwork(object):
    ## Framework for implementing neural networks in tensorflow
    #
    # @param n_inputs: (int) number of inputs into network
    # @param n_outputs: (int) number of outputs into network
    # @param batch_size: (int) number of inputs into network
    # @param learning_rate: (float) learning rate
    # @param steps: (int) number of training epochs
    # @param model_name: (str) name of model
    # @param print_steps: (int) print loss every n-th epoch
    # @param n_save_steps:  (int) absolute number of save steps
    # @param cyclic_steps:  (int) call cyclic method every n-th epoch
    # @param path:  (str) save path
    # @param fold:  (int) fold number
    # @param optimizer_name: (str) name of the desired optimizer
    # @param grad_clip: DON'T USE!
    # @param save:  (bool) save the weights
    # @param seed (int):  seed used for initialisation of networks weights
    # @param mutex (threading.Lock()):  needed if threading backend is used (use_shared_memory = True)
    # and non thread safe methods are used (e. g., matplotlib functions)
    # @param use_shared_memory (bool):  whether or not to use shared memory (changes gpu placement method)
    def __init__(self, n_inputs=0, n_outputs=0, batch_size=None, learning_rate=0.1, model_name='',
                 path='/saved_models', fold=0, optimizer_name='RMSProp', grad_clip=None, save=False, seed=0,
                 mutex=None, use_shared_memory=False):

        self.n_inputs = n_inputs  # for _create_IO
        self.n_outputs = n_outputs  # for _create_IO
        self.batch_size = batch_size  # for _fit and predict
        self.learning_rate = learning_rate  # for fit
        self.model_name = model_name  # for saver
        self.fold = fold
        self.optimizer_name = optimizer_name
        self.grad_clip = grad_clip
        self.save = save

        # check if any self object is set in the inherenced model, that's not an argument of the __init__
        args, _, _, _ = inspect.getargspec(self.__init__)
        basic_args, _, _, _ = inspect.getargspec(BasicNetwork.__init__)
        for key in self.__dict__:
            if key not in args and (key not in basic_args):
                raise ValueError('undefined parameter "' + key + '" in network')

        hyperparams = self.__dict__
        self.basic_params = []

        # remove the nasty hyperparam element
        param_dict = dict(filter(lambda x: x[0] != 'hyperparams', hyperparams.items()))

        for params in param_dict:
            self.basic_params.append(params)

        self.seed = seed  # network init seed for reproducable
        self.path = path  # for saver

        self.mean_loss = 1e10
        self.session_acitve = False
        self.score_dict = {}
        self.score_lst = []
        self.valid_loss_lst = []
        self.train_score_lst = []
        self.epoch_lst = []

        self.scores = {}

        self.start_epoch = 1
        self.resume_fit = False

        # detect overwritten methods
        self.overwritten_methods = detect_overridden(BasicNetwork, self)

        self.mutex = mutex

        self.used_gpu = 0
        self.use_shared_memory = use_shared_memory

        # sleep for one second to enable save network spawn
        time.sleep(1)

    ##  Get parameter names for the estimator
    #   arguments of the constructor are the arguments of the estimator
    @classmethod
    def _get_param_names(self):
        # Get all arguments of the constructor
        args_super, _, _, _ = inspect.getargspec(self.__init__)
        args_basic, _, _, _ = inspect.getargspec(BasicNetwork.__init__)
        args = sorted(set([p for p in args_super if p != 'self'] + [p for p in args_basic if p != 'self']))

        # remove self from the arguments
        return args

    ##  Get parameters for this estimator
    #   @return: (dict)  parameters and the set values of the current estimator
    def get_params(self):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            out[key] = value
        return out

    ##  Set the parameters of this estimator.
    #   @return: self
    #   @raise: (ValueError) if you try to set a parameter that is not suitable for the estimator
    def set_params(self, **params):

        if not params:
            # Simple optimization to gain speed
            return self

        valid_params = self.get_params()

        nested_params = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    ##  get a list of scores
    #   @return                    list of scores
    def get_score_dict(self):
        return [remove_addresses(str(self.get_params())), self.get_params(), self.score_lst, self.epoch_lst,
                self.train_score_lst]

    ##  core fit method
    #   @param X:                   input training data
    #   @param y:                   ground truth  training data
    #   @param train:               indices of train-set X[train], y[train]
    #   @param validation:          indices of valid-set X[validation], y[validation]
    #   @param scoring: (bool)      option for validate during training
    #   @param steps:   (int)       number of train steps
    #   @param print_steps: (int)   print loss every n'th train step
    #   @param score_steps: (int)   scores every n'th train step (calls score(X[validation], y[validation]) method)
    #   @param cyclic_steps:(int)   calls cyclic((X[validation], y[validation]) every n'th train step
    #   @param **kwargs             variable number of keyword arguments
    def fit(self, X, y=None, train=None, validation=None, scoring=False, steps=1, print_steps=None, score_steps=None,
            cyclic_steps=None, **kwargs):

        # check if scoring is possible
        if not isinstance(validation, type(None)):
            valid_data_exists = True
        else:
            valid_data_exists = False

        if scoring and not valid_data_exists:
            print('fold_' + str(self.fold), "WARNING: Cant score without validation data!")
            scoring = False

        # split into train and test data
        if valid_data_exists == True and y is not None:
            X_train = X[train]
            y_train = y[train]
            X_valid = X[validation]
            y_valid = y[validation]
        elif valid_data_exists == True and y is None:
            X_train = X[train]
            X_valid = X[validation]
            y_train = None
            y_valid = None
        else:
            X_train = X
            y_train = y
            X_valid = None
            y_valid = None

        print('fold_' + str(self.fold), "Train the model for %s epochs" % steps)

        if cyclic_steps:
            print('fold_' + str(self.fold), "Call Cyclic() on validation data every %s'th epoch" % cyclic_steps)
        else:
            cyclic_steps = steps * 3

        if print_steps:
            print('fold_' + str(self.fold), "Prints current intermediate train loss every %s'th epoch" % print_steps)
        else:
            print_steps = steps * 3

        if score_steps and scoring == True:
            print('fold_' + str(self.fold), "Score the model on validation data every %s'th epoch" % score_steps)
            # check if user defined scoring method is found
            if 'get_score' not in self.overwritten_methods:
                print(
                    'fold_' + str(self.fold), "WARNING: No user defined scoring method is found, using 1/SSE as score!")

        if score_steps and scoring == False and valid_data_exists == True:
            print('fold_' + str(self.fold), "Computes loss on validation data every %s'th epoch" % score_steps)
        if score_steps == None:
            score_steps = steps * 3

        if self.save == True:
            print('fold_' + str(self.fold), "Save the model every %s'th epoch" % score_steps)

        # Build Tensorflow Graph
        if not self.resume_fit:
            self.build_graph()

        if not os.path.exists(self.param_path):
            os.makedirs(self.param_path)

        # load the current hyperparams
        hyperparams = self.__dict__
        # remove the nasty hyperparam element
        param_dict = dict(filter(lambda x: x[0] != 'hyperparams', hyperparams.items()))
        # only use the core hyperparams (self.__dict__ returns all self objects)
        core_param_dict = {}
        for param in self.basic_params:
            tmp = dict(filter(lambda x: x[0] == param, hyperparams.items()))
            core_param_dict.update(tmp)

        hyperparam_file = open(self.param_path + "/hyperparameter.pkl", "wb")
        pickle.dump(core_param_dict, hyperparam_file)
        hyperparam_file.flush()
        os.fsync(hyperparam_file)

        # Tensorboard Utils
        train_loss_ = tf.placeholder(tf.float64)
        valid_loss_ = tf.placeholder(tf.float64)
        score_ = tf.placeholder(tf.float64)

        # Tensor board file writer
        self.file_writer = tf.summary.FileWriter(self.param_path + '/train',
                                                 self.sess.graph)
        for param in param_dict['basic_params']:
            summary_tf = tf.convert_to_tensor(str(param_dict[param]))
            summary_text_op = tf.summary.text(str(param), summary_tf)
            summary, summary_out = self.sess.run([summary_tf, summary_text_op])
            self.file_writer.add_summary(summary_out)

        train_loss_summary_ = tf.summary.scalar('train_loss', train_loss_)
        valid_loss_summary_ = tf.summary.scalar('valid_loss', valid_loss_)
        valid_score_summary_ = tf.summary.scalar('valid_score', score_)

        # Train the model

        # fit loop
        epoch = self.start_epoch
        train_range = range(self.start_epoch, steps + 1)
        for epoch in train_range:

            # get loss
            curr_loss = self.fit_epoch(X_train, y_train, **kwargs)
            train_loss, train_loss_summary = self.sess.run([train_loss_, train_loss_summary_],
                                                           feed_dict={train_loss_: curr_loss})
            self.file_writer.add_summary(train_loss_summary, global_step=epoch)

            if epoch % print_steps == 0:
                print(self.fold, "step:", epoch, "loss:", curr_loss, self.name)
            # if the current epoch is a scoring epoch or the last epoch, do the scoring
            if (epoch % score_steps == 0) or epoch == steps:
                if valid_data_exists:
                    valid_score = '--'
                    if scoring:
                        # build and save valid score
                        valid_score = self.score(X_valid, y_valid, step=epoch, **kwargs)
                        score_summary_out = self.sess.run(valid_score_summary_,
                                                          feed_dict={score_: valid_score})

                        self.file_writer.add_summary(score_summary_out, global_step=epoch)

                    curr_valid_loss = self.calulate_loss(X_valid, y_valid, **kwargs)

                    valid_loss_summary = self.sess.run(valid_loss_summary_, feed_dict={valid_loss_: curr_valid_loss})
                    self.file_writer.add_summary(valid_loss_summary, global_step=epoch)

                    print(self.fold, "validation: step:", epoch, "loss:", curr_valid_loss,
                          "score:", valid_score, self.name)

                    self.save_backup(epoch, valid_score)

                else:
                    self.save_backup(epoch)

                if self.save:
                    # save the weights and biases of the model (tf variables) in a checkpoint-file
                    self.saver.save(self.sess, os.path.join(self.path, self.hash_name, str(epoch), 'model.ckpt'))
                    # save the current epoch in a pickle-file
                    with open(os.path.join(self.param_path, 'last_epoch.pkl'), 'wb') as epoch_file:
                        # dump the file
                        pickle.dump(str(epoch), epoch_file)
                        # check that the file is properly synced on disk
                        os.fsync(epoch_file)

            if epoch % cyclic_steps == 0 and valid_data_exists == True:
                self.cyclic(X=X_valid, y=y_valid, step=epoch, **kwargs)

        # Save last step

        # save the current epoch in a pickle-file
        with open(os.path.join(self.param_path, 'last_epoch.pkl'), 'wb') as epoch_file:
            pickle.dump(str(epoch), epoch_file)
            # check that the file is properly synced on disk
            os.fsync(epoch_file)

    ## creates a backup of the model
    # - saves weights and biases of the model (tf variables) in a checkpoint-file
    # - saves epoches and scores in a pickle-file while maintaining the previous epoches and scores
    # @param epoch: int epoch to be saved
    # @param valid_score:   score to be saved
    def save_backup(self, epoch, valid_score=None):
        # save last checkpoint, overwrites the previously saved checkpoint
        self.saver.save(self.sess, os.path.join(self.path, self.hash_name, 'backup', 'model.ckpt'))
        with open(os.path.join(self.path, self.hash_name, 'backup', 'epoch.pkl'), 'wb') as epoch_file:
            pickle.dump(str(epoch), epoch_file)

        # save the score
        if not isinstance(valid_score, type(None)):
            # score-file where the epochs and the
            score_file = os.path.join(self.path, self.hash_name, 'backup', 'scores.pkl')
            #
            if not os.path.isfile(score_file):
                # init dict with the epoch as key and the score as value
                scores = {epoch: valid_score}
                with open(score_file, 'wb') as s_file:
                    # dump the dict to a pickle file
                    pickle.dump(scores, s_file)
                    # check that the file is properly synced on disk
                    os.fsync(s_file)
            else:
                with open(score_file, 'rb') as s_file:
                    # check that the file is properly synced on disk before loading
                    os.fsync(s_file)
                    # load previous epoches and scores
                    scores = pickle.load(s_file)
                with open(score_file, 'wb') as s_file:
                    # update dict
                    scores.update({epoch: valid_score})
                    # dump the updated dict
                    pickle.dump(scores, s_file)
                    # check that the file is properly synced on disk
                    os.fsync(s_file)

    ## continues to fit by loading the backup files
    #   @param X:                   input training data
    #   @param y:                   ground truth  training data
    #   @param train:               indices of train-set X[train], y[train]
    #   @param validation:          indices of valid-set X[validation], y[validation]
    #   @param scoring: (bool)      option for validate during training
    #   @param steps:   (int)       number of train steps
    #   @param print_steps: (int)   print loss every n'th train step
    #   @param score_steps: (int)   scores every n'th train step (calls score(X[validation], y[validation]) method)
    #   @param cyclic_steps:(int)   calls cyclic((X[validation], y[validation]) every n'th train step
    #   @param path_to_canceled:(str)   optional path to canceled model
    #   @param **kwargs:            variable number of keyword arguments
    def continue_fit(self, X, y, train=None, validation=None, scoring=False, steps=1, print_steps=None,
                     score_steps=None, cyclic_steps=None, path_to_canceled=None, **kwargs):

        if not isinstance(path_to_canceled, type(None)):
            self.load(path_to_canceled, epoch=None)

        else:
            print('WARNING: could not load the stored model, retrain the model from scratch!')
            print('WARNING: Make sure that the parameters are set.')

        self.fit(X, y, train=train, validation=validation, scoring=scoring, steps=steps, print_steps=print_steps,
                 score_steps=score_steps, cyclic_steps=cyclic_steps, **kwargs)

    ##  Batch generator that yields at the currently active data batch
    # @param X (list, np.array, or shm.array): input data for batch generation
    # @param y (list, np.array, shm.array, or None): ground truth data for batch generation (ignored if None)
    # @param batch_size (int or None): batch size, None for full batch
    # @param expect_x_and_y (bool): has to be true if y is None, but two outputs are expected (unsupervised tasks)
    # @param **kwargs: variable number of keyword arguments
    # @yields: the next batch
    @staticmethod
    def process_as_batch(X, y=None, batch_size=None, expect_x_and_y=False, **kwargs):
        # Check which types of arrays are used and if y is expected but is None (this is the case for unsupervised task)
        #  both X and y are passed and are list or np.array
        if isinstance(y, np.ndarray) or isinstance(y, list):
            # use batch processing
            if batch_size and (len(X) // batch_size) != 0:
                # compute complete batches
                batch = 0
                for batch in range(0, len(X) // batch_size):
                    batch_X = X[batch * batch_size:batch * batch_size + batch_size]
                    batch_y = y[batch * batch_size:batch * batch_size + batch_size]
                    yield batch_X, batch_y

                # compute rest
                batch_X = X[batch * batch_size + batch_size:]
                batch_y = y[batch * batch_size + batch_size:]
                yield batch_X, batch_y
            # use full batch
            else:
                yield X, y
        # both X and y are passed and are shm.array
        elif isinstance(X, shm.array) and isinstance(y, shm.array):
            # use batch processing
            if batch_size and (len(X) // batch_size) != 0:
                # compute complete batches
                batch = 0
                for batch in range(0, len(X) // batch_size):
                    batch_X = X.get_copy(slice(batch * batch_size, batch * batch_size + batch_size))
                    batch_y = y.get_copy(slice(batch * batch_size, batch * batch_size + batch_size))
                    yield batch_X, batch_y

                # compute rest
                batch_X = X.get_copy(slice(batch * batch_size + batch_size, None))
                batch_y = y.get_copy(slice(batch * batch_size + batch_size, None))
                yield batch_X, batch_y
            # use full batch
            else:
                yield X.get_copy(), y.get_copy()
        # only X is passed, y is None, and y is not expected (only one output is expected)
        elif y is None and not expect_x_and_y:
            # X is shm.array
            if isinstance(X, shm.array):
                # use batch processing
                if batch_size and (len(X) // batch_size) != 0:
                    # compute complete batches
                    batch = 0
                    for batch in range(0, len(X) // batch_size):
                        batch_X = X.get_copy(slice(batch * batch_size, batch * batch_size + batch_size))
                        yield batch_X
                    # compute rest
                    batch_X = X.get_copy(slice(batch * batch_size + batch_size, len(X)))
                    yield batch_X
                # use full batch
                else:
                    yield X
            # X is list or np.array
            else:
                # use batch processing
                if batch_size and (len(X) // batch_size) != 0:
                    # compute complete batches
                    batch = 0
                    for batch in range(0, len(X) // batch_size):
                        batch_X = X[batch * batch_size:batch * batch_size + batch_size]
                        yield batch_X
                    # compute rest
                    batch_X = X[batch * batch_size + batch_size:]
                    yield batch_X
                # use full batch
                else:
                    yield X
        # only X is passed, y is None, and y is expected (two outputs are expected, y = X, unsupervised tasks)
        else:
            # X is shm.array
            if isinstance(X, shm.array):
                # use batch processing
                if batch_size and (len(X) // batch_size) != 0:
                    # compute complete batches
                    batch = 0
                    for batch in range(0, len(X) // batch_size):
                        batch_X = X.get_copy(slice(batch * batch_size, batch * batch_size + batch_size))
                        yield batch_X, batch_X
                    # compute rest
                    batch_X = X.get_copy(slice(batch * batch_size + batch_size, len(X)))
                    yield batch_X, batch_X
                # use full batch
                else:
                    output = X.get_copy()
                    yield output, output
            # X is list or np.array
            else:
                if batch_size and (len(X) // batch_size) != 0:
                    # compute complete batches
                    batch = 0
                    for batch in range(0, len(X) // batch_size):
                        batch_X = X[batch * batch_size:batch * batch_size + batch_size]
                        yield batch_X, batch_X
                    # compute rest
                    batch_X = X[batch * batch_size + batch_size:]
                    yield batch_X, batch_X
                # use full batch
                else:
                    yield X, X

    ##  Shuffels the input and ground truth data over the first dimention
    #   @param X:   input data
    #   @param y:   ground truth data
    #   @param **kwargs: variable number of keyword arguments
    def shuffle(self, X, y=None, **kwargs):
        y_valid = not isinstance(y, type(None))
        import random
        # create an empty array
        Xs = np.zeros_like(X)
        # get list of indexes
        lst = list(range(X.shape[0]))
        if y_valid:
            ys = np.zeros_like(y)
        # shuffle the index list
        random.shuffle(lst)
        # loop over index list and fill the empty arrays
        for idx, shuffled_idx in enumerate(lst):
            Xs[idx] = X[shuffled_idx]
            if y_valid:
                ys[idx] = y[shuffled_idx]
        return Xs, ys

    ## fits data for one full epoch (in iterations if batched)
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    def fit_epoch(self, X, y, **kwargs):
        # shuffel the data
        # X, y = shuffle(X, y, **kwargs)

        loss = []

        # fit data in batches if self.batch_size is not None
        # iteration loop
        for batch_X, batch_y in BasicNetwork.process_as_batch(X=X, y=y, batch_size=self.batch_size,
                                                              expect_x_and_y=True, **kwargs):
            # check if list contains items
            if len(batch_X) > 0:
                _, batch_loss = self.sess.run([self.optimizer, self.loss_fkt],
                                              feed_dict={self.input: batch_X, self.output: batch_y,
                                                         self.is_train: True})
                loss.append(batch_loss)

        cur_loss = np.mean(np.array(loss))
        return cur_loss

    ##  calculates loss (optional in batches if self.batch_size is not None)
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    the loss for the data
    def calulate_loss(self, X, y, **kwargs):
        loss = []
        for batch_X, batch_y in BasicNetwork.process_as_batch(X=X, y=y, batch_size=self.batch_size,
                                                              expect_x_and_y=True, **kwargs):
            if len(batch_X) > 0:
                batch_loss = self.sess.run(self.loss_fkt,
                                           feed_dict={self.input: batch_X, self.output: batch_y,
                                                      self.is_train: False})
                loss.append(batch_loss)

        cur_loss = np.mean(np.array(loss))
        return cur_loss

    ## runs forward graph (optional in batches if self.batch_size is not None)
    #   @param X:                   input data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    the prediction
    def predict(self, X, **kwargs):
        output = []
        for batch_X in BasicNetwork.process_as_batch(X=X, batch_size=self.batch_size,
                                                     **kwargs):
            if len(batch_X) > 0:
                pred = self.sess.run(self.prediction, feed_dict={self.input: batch_X})
                output += pred.tolist()
        return output

    ## computes score (optional in batches if self.batch_size is not None)
    #   overload _get_score for your needs
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    score
    def score(self, X, y, step=None, **kwargs):
        cur_score = []
        for batch_X, batch_y in BasicNetwork.process_as_batch(X=X, y=y, batch_size=self.batch_size,
                                                              expect_x_and_y=True, **kwargs):
            if len(batch_X) > 0:
                cur_score.append(self.get_score(batch_X, batch_y, **kwargs))
        return np.mean(np.array(cur_score))

    ##  This method is cyclic called on validation data every n'th train epoch
    #   Overload this method for your needs
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    def cyclic(self, X, y, step=None, **kwargs):

        pass

    ##  Loads a saved model from given path with specified epoch
    #   @param path:  (str) path to saved model
    #   @param epoch: (int) epoch to be loaded
    def load(self, path, epoch=None):

        hyper_param_path = os.path.join(path, 'hyperparameter.pkl')

        if os.path.isfile(hyper_param_path):
            # load parameters from pickle file 
            parameters = pickle.load(open(hyper_param_path, 'rb'))
            # remove 'basic_params from param dict
            if 'basic_params' in parameters:
                del parameters['basic_params']

            # set parameters
            self.set_params(**parameters)
            # call build graph
            self.build_graph()
        if not epoch:
            save_last = os.path.join(path, 'backup', 'model.ckpt')
            f_epoch = open(os.path.join(path, 'backup', 'epoch.pkl'), 'rb')
            epoch = int(pickle.load(f_epoch, encoding='latin1'))
        else:
            save_last = os.path.join(path, str(epoch), 'model.ckpt')
        print('Load model')
        self.saver.restore(self.sess, save_last)
        print('restored:', save_last)
        self.start_epoch = epoch + 1
        self.resume_fit = True

    ##  The overloadable score method
    #   This method is called within score, overload this method with your costume score
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    score
    def get_score(self, X, y, **kwargs):
        predictions = self.predict(X, **kwargs)
        error = 0
        for i, prediction in enumerate(predictions):
            error += np.sum(np.square(np.array(prediction) - np.array(y[i])))
        if i != 0.0:
            error /= i
        return error

    ##  The overloadable network architecture
    #   define your tensorflow graph here
    #   inputs are encoded within the placeholder self.input
    #   output of the forward graph is within self.prediction
    def run_graph(self):
        # simple linear layer
        self.prediction = tf.layers.dense(self.input, self.n_outputs, activation=None)

    ##  Define your tensorflow loss here
    #   inputs = self.prediction
    #   outputs (ground truth) are encoded within the placeholder self.outputs
    def get_loss(self):
        self.loss_fkt = tf.reduce_mean(tf.square(tf.subtract(self.prediction, self.output)))

    ##  Defines the optimizer used for the training
    #   select one of the with self.optimizer_name
    #   ['Adam','RMSProp','GradientDescent','Nesterov','Momentum']
    def define_optimizer(self):
        if self.optimizer_name == 'Adam':
            optimizer_def = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'RMSProp':
            optimizer_def = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'GradientDescent':
            optimizer_def = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Nesterov':
            optimizer_def = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                       use_nesterov=True)
        elif self.optimizer_name == 'Momentum':
            optimizer_def = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        else:
            optimizer_def = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # Gradient Clipping
        if self.grad_clip:
            with tf.name_scope("Gradient_Clipping"):
                grads_and_vars = optimizer_def.compute_gradients(self.loss_fkt)
                clipped_grads = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in
                                 grads_and_vars]
                self.optimizer = optimizer_def.apply_gradients(clipped_grads)
        else:
            self.optimizer = optimizer_def.minimize(self.loss_fkt)
        print('Optimizer ' + self.optimizer_name + ' Set')

    ## This method is used to define the input and output placholders which are feet into the tensorflow graph
    #
    def create_IO(self):
        self.input = tf.placeholder(tf.float32, [None, self.n_inputs], name='X')
        self.output = tf.placeholder(tf.float32, [None, self.n_outputs], name='y')

    ##  This method is used to build the tensorflow graph
    #   the tensorflow graph is dynamically placed on the most suitable GPU
    def build_graph(self):
        """Doxygen"""
        # Reset default graph
        tf.reset_default_graph()
        # set tf random seed to reproduce results

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        if not self.use_shared_memory:
            # Sort GPUs by available memory
            if os.environ["CUDA_VISIBLE_DEVICES"]:
                available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                available_devices = '0'
            used_gpu = sort_gpus(name='My devices', AVAILABLE_DEVICES=available_devices)
            mask_gpus(used_gpu[:1])
        else:
            # get which gpu to use
            ppid = os.getppid()
            filename = str(ppid) + "_gpu_processes"

            if not os.path.exists(filename):
                # create a file with name of parent pid as name, init [0]*NUMBER_OF_GPUS_USED
                # get parent pid (same in every subprocess or thread)
                print("ppid", ppid)
                # get used gpus or default to 0
                if os.environ["CUDA_VISIBLE_DEVICES"]:
                    available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    available_devices = '0'

                available_devices = available_devices.split(",")
                # save to with distinct filename (unique by use of ppid)
                json.dump([0] * len(available_devices), open(str(ppid) + "_gpu_processes", "w"))

            lock = FileLock(filename + ".lock")
            with lock:
                processes_on_gpus = json.load(open(filename, 'r'))
                self.used_gpu = np.argmin(processes_on_gpus)
                processes_on_gpus[self.used_gpu] += 1
                json.dump(processes_on_gpus, open(filename, 'w'))

        new_dict = dict(self.get_params())
        new_dict.pop('path', None)

        # create output string
        # first remove all adresses
        model_output_name = remove_addresses(str(new_dict))
        # then remove all unnecessary strings and special charachters except '_' and ','
        special_characters = [".", "{", "}", "[", "]", ":", "<", ">", "BasicDense", "model_name", "'", "function",
                              " at ",
                              "_name"]
        for char in special_characters:
            model_output_name = model_output_name.replace(char, "")
        model_output_name = model_output_name.replace(", ", "_")
        model_output_name = model_output_name.replace(" ", "_")
        model_output_name = model_output_name.replace("__", "_")
        special_words = ["grad_clip_None_",
                         "_mutex_threadlock_object",
                         "_mutex_None",
                         "_use_shared_memory_True",
                         "_use_shared_memory_False",
                         "_save_True",
                         "_save_False",
                         ]
        for word in special_words:
            model_output_name = model_output_name.replace(word, "")

        self.name = model_output_name
        # hash the name if it's too long, since the name is part of the save
        if len(self.name) > 200:
            self.hash_name = hashlib.sha256(self.name.encode('utf-8')).hexdigest()
        else:
            self.hash_name = self.name

        self.param_path = os.path.join(self.path, self.hash_name)

        self.save_last = os.path.join(self.path, self.hash_name, "model.ckpt")

        # is_train is necessary for layers that are only active during the training
        self.is_train = tf.placeholder_with_default(False, shape=(), name='is_train')

        # define place holders
        self.create_IO()

        if self.use_shared_memory:
            used_gpu = '/device:GPU:' + str(self.used_gpu)
            print("Used gpu", used_gpu)
            with tf.device(used_gpu):
                # define run graph
                self.run_graph()

                # Loss function
                self.get_loss()

                # Optimizer
                self.define_optimizer()
        else:
            # define run graph
            self.run_graph()

            # Loss function
            self.get_loss()

            # Optimizer
            self.define_optimizer()

        # Create tf session
        # GPU options
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        init = tf.global_variables_initializer()

        # from tensorflow.python import debug as tf_debug
        self.sess = tf.Session(config=config)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=0)

        # Initilize Session
        self.sess.run(init)
        self.session_acitve = True
