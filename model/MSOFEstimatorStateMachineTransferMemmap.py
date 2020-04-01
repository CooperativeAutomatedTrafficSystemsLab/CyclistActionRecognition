from LF_Python3.basic_network import BasicNetwork
from LF_Python3.layers import hidden_architecture
import tensorflow as tf
import cv2 as cv
import random
from tensorflow.contrib.layers import flatten
import numpy as np
from multiprocessing import Queue, Process
import gc
import os
from matplotlib import pyplot as plt
import pickle
import zlib
from sklearn.metrics import f1_score
from BasicMovementDetectionEvaluation.metrics.create_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import brier_score_loss
from model.InceptionI3D import InceptionI3D
import hashlib
from BasicMovementDetectionEvaluation.metrics.create_qq_plot import create_qq_plot
from LF_Python3.tf_memory import sort_gpus, mask_gpus
from LF_Python3.utils import remove_addresses
from filelock import FileLock

n_validation_steps = 50


## Class containing a parameterizable tensorflow implementation of a basic movement detection based on motion sequences
#
class MSOFEstimatorStateMachine(BasicNetwork):
    num_scoring_batches = 50

    ## Constructor
    #
    # @param: cnn_architecture (str): name of ConvNet architecture to use (so for only InceptionI3D implemented)
    # @param: cnn_params (dict): dictionary containing additional cnn parameters
    # @param: use_ms_hk1 (bool): if True, image sequence of camera 1 is used
    # @param: use_ms_hk2 (bool): if True, image sequence of camera 2 is used
    # @param: use_of_hk1 (bool): if True, of sequence of camera 1 is used
    # @param: use_of_hk2 (bool): if True, of sequence of camera 2 is used
    # @param: use_trajectory (bool): if True, trajectory is used
    # @param: hidden_architecture_trajectory (dict): dictionary containing hidden architecture config
    # e.g., {'FCL': {'activation': tf.nn.relu, 'layers': [10] * 3, 'keep_prop': 1.0}}
    # @param: hidden_architecture_feature_union (dict): dictionary containing hidden architecture config
    # e.g., {'FCL': {'activation': tf.nn.relu, 'layers': [10] * 3, 'keep_prop': 1.0}}
    # @param: classifcation_net_activation (tf activation): activation used in classification net
    # @param: train_wait_motion (bool):  if True, wait_motion state machine is trained
    # @param: train_turn_straight (bool):  if True, turn_straight state machine is trained
    # @param: train_left_right (bool):  if True, left_right state machine is trained
    # @param: train_start_stop_move (bool): if True, start_stop_move state machine is trained
    # @param: ms_size (list): size of input image sequences, e.g., [192, 192, 10]
    # @param: of_size (list): size of input of sequences, e.g., [192, 192, 18]
    # @param: trajectory_input_size (list): length and dims of input trajectory, e.g., [50, 3]
    # @param: learning_rate (float): learning rate used in optimizer
    # @param: model_name (str): name to be save in front of configuration string
    # @param: fold (int): used validation fold (only saved in string)
    # @param: path (str): path where all results are save
    # @param: batch_size (int): size of training batches
    # @param: optimizer_name (str): string containing optimizer name, e.g. 'Adam'
    # @param: balance_samples (bool): if True, classes are balanced during training
    # @param: kp (bool, float): keep probability for dropout, if False, no Dropout is performed
    # @param: save (bool): if True, network is saved every validation step
    # @param: mutex (Threadding.Lock()): need for matplotlib functions if threading (not multiprocessing) is performed
    # @param: use_shared_memory (bool): if True, threading backend is used
    def __init__(self, cnn_architecture='InceptionI3D',
                 cnn_params={'reduction_ratio': 4, 'inception_resnet_arch': [5, 10, 5]},
                 use_ms_hk1=False, use_ms_hk2=False, use_of_hk1=True, use_of_hk2=True, use_trajectory=True,
                 hidden_architecture_trajectory={
                     'FCL': {'activation': tf.nn.relu, 'layers': [10] * 3, 'keep_prop': 1.0}},
                 hidden_architecture_feature_union={
                     'FCL': {'activation': tf.nn.relu, 'layers': [100] * 3, 'keep_prop': 1.0}},
                 classifcation_net_activation=tf.nn.tanh,
                 train_wait_motion=True,
                 train_turn_straight=False,
                 train_left_right=False,
                 train_start_stop_move=False,
                 ms_size=[192, 192, 10],
                 of_size=[192, 192, 18],
                 trajectory_input_size=[50, 3],
                 learning_rate=0.000001,
                 model_name='MSOFClassifier',
                 fold=0,
                 path='~/saved_models/',
                 batch_size=2,
                 optimizer_name='Adam',
                 balance_samples=False,
                 kp=False,
                 save=False,
                 mutex=None,
                 use_shared_memory=False):

        self.cnn_architecture = cnn_architecture
        self.cnn_params = cnn_params
        self.use_ms_hk1 = use_ms_hk1
        self.use_ms_hk2 = use_ms_hk2
        self.use_of_hk1 = use_of_hk1
        self.use_of_hk2 = use_of_hk2
        self.use_trajectory = use_trajectory
        self.hidden_architecture_trajectory = hidden_architecture_trajectory
        self.hidden_architecture_feature_union = hidden_architecture_feature_union
        self.classifcation_net_activation = classifcation_net_activation
        self.batch_size = batch_size
        self.ms_size = ms_size
        self.of_size = of_size
        self.trajectory_input_size = trajectory_input_size
        self.train_wait_motion = train_wait_motion
        self.train_turn_straight = train_turn_straight
        self.train_left_right = train_left_right
        self.train_start_stop_move = train_start_stop_move
        self.balance_samples = balance_samples

        # keep Prop, for dropout
        # check for illegal Values
        if isinstance(kp, float):
            if kp > 0.0 and kp <= 1.0:
                self.kp = kp
            else:
                self.kp = False
        else:
            self.kp = False

        """Set basic Hyperparams in Superclass"""
        super(MSOFEstimatorStateMachine, self).__init__(batch_size=batch_size, learning_rate=learning_rate,
                                                        model_name=model_name,
                                                        path=path, fold=fold, optimizer_name=optimizer_name,
                                                        save=save, use_shared_memory=use_shared_memory)

        self.mutex = mutex
        self.buffer_size = 50

    ## Extracts compressed image sequences from dataset and stores them in train buffer
    #
    # @param: X (list(dict)): list of dictionaries containing compressed network inputs
    # @param: y (list(dict)): list of dictionaries containing network outputs
    # @param: q (Multiprocesing.Queue()): buffer to be loaded
    # @param: state_machine (str): state machine labels to extract from y
    # @param: balance_samples (bool): if True, classes are balances
    def load_buffer(self, X, y, q, state_machine='wait_motion', balance_samples=True):
        if state_machine == 'wait_motion':
            n_samples = [0, 0]
            key = 'labels_wait_motion'
        elif state_machine == 'turn_straight':
            n_samples = [0, 0]
            key = 'labels_turn_straight'
        elif state_machine == 'start_stop_move':
            n_samples = [0, 0, 0]
            key = 'labels_start_stop_move'
        else:
            n_samples = [0, 0]
            key = 'labels_left_right'

        class_inidices = [[] for _ in range(len(n_samples))]
        for i in range(len(y)):
            if sum(y[i][key]) == 1:
                class_inidices[np.argmax(y[i][key])].append(i)

        while self.buffering:
            # get random samples from data
            sample_X = []
            sample_y = []

            if balance_samples:
                # find equal number of samples
                max_samples = int(self.batch_size / len(n_samples))
                sample_idx = []
                samples_all_classes = []
                for class_index in class_inidices:
                    # concatenate all used classes for final sampling
                    samples_all_classes += class_index
                    # add samples of one specific class
                    sample_idx += random.sample(class_index, max_samples)
                rest = self.batch_size - len(sample_idx)
                if rest > 0:
                    # final sampling
                    sample_idx += random.sample(samples_all_classes, self.batch_size - len(sample_idx))
                random.shuffle(sample_idx)
            else:
                # make sure that at least 'min_samples' samples of every class are present
                min_samples = 1
                sample_idx = []
                samples_all_classes = []
                for class_index in class_inidices:
                    # concatenate all used classes for final sampling
                    samples_all_classes += class_index
                    # add samples of one specific class
                    sample_idx += random.sample(class_index, min_samples)
                # final sampling
                sample_idx += random.sample(samples_all_classes, self.batch_size - len(n_samples) * min_samples)

            for sample_id in sample_idx:
                sample_X.append(pickle.loads(X[sample_id]))
                sample_y.append(y[sample_id])

            output_batch = []
            for i in range(len(sample_X)):
                X_dict = {}
                if self.use_ms_hk1:
                    ms_hk1 = [cv.cvtColor(cv.imdecode(image, 1), cv.COLOR_BGR2GRAY) for image in sample_X[i]['ms_hk1']]
                    X_dict.update({'ms_hk1': ms_hk1})
                if self.use_ms_hk2:
                    ms_hk2 = [cv.cvtColor(cv.imdecode(image, 1), cv.COLOR_BGR2GRAY) for image in sample_X[i]['ms_hk2']]
                    X_dict.update({'ms_hk2': ms_hk2})
                if self.use_of_hk1:
                    of_hk1 = pickle.loads(zlib.decompress(sample_X[i]['of_hk1'])).astype(np.float16) * 2.0 / 255.0 - 1.0
                    of_hk1 = np.concatenate(of_hk1, axis=-1)
                    X_dict.update({'of_hk1': of_hk1})
                if self.use_of_hk2:
                    of_hk2 = pickle.loads(zlib.decompress(sample_X[i]['of_hk2'])).astype(np.float16) * 2.0 / 255.0 - 1.0
                    of_hk2 = np.concatenate(of_hk2, axis=-1)
                    X_dict.update({'of_hk2': of_hk2})
                if self.use_trajectory:
                    trajectory = sample_X[i]['trajectory']
                    X_dict.update({'trajectory': trajectory})
                output_batch.append({"X": X_dict, 'y': sample_y[i]})

            q.put(output_batch)
        print('thread stopped')

    ## This method is used to define the input and output placholders which are feet into the tensorflow graph
    #
    def create_IO(self):
        self.input_ms_hk1 = tf.placeholder(tf.float32, [None, self.ms_size[0], self.ms_size[1], self.ms_size[2]],
                                           name='input_ms_hk1')
        self.input_ms_hk2 = tf.placeholder(tf.float32, [None, self.ms_size[0], self.ms_size[1], self.ms_size[2]],
                                           name='input_ms_hk2')
        self.input_of_hk1 = tf.placeholder(tf.float32, [None, self.of_size[0], self.of_size[1], self.of_size[2]],
                                           name='input_of_hk1')
        self.input_of_hk2 = tf.placeholder(tf.float32, [None, self.of_size[0], self.of_size[1], self.of_size[2]],
                                           name='input_of_hk2')
        self.input_trajectory = tf.placeholder(tf.float32,
                                               [None, self.trajectory_input_size[0], self.trajectory_input_size[1]],
                                               name="input_trajectory")
        self.labels_wait_motion = tf.placeholder(tf.float32, [None, 2], name='labels_wait_motion')
        self.labels_turn_straight = tf.placeholder(tf.float32, [None, 2], name='labels_turn_straight')
        self.labels_start_stop_move = tf.placeholder(tf.float32, [None, 3], name='labels_start_stop_move')
        self.labels_left_right = tf.placeholder(tf.float32, [None, 2], name='labels_left_right')
        self.training_flag = tf.placeholder(tf.bool, name='training_flag')

        # tensorboard
        self._scalar = tf.placeholder(tf.float32)
        self.summary_f1_wait_motion_micro = tf.summary.scalar('f1_wait_motion_micro', self._scalar)
        self.summary_f1_wait_motion_macro = tf.summary.scalar('f1_wait_motion_macro', self._scalar)
        self.summary_f1_turn_straight_micro = tf.summary.scalar('f1_turn_straight_micro', self._scalar)
        self.summary_f1_turn_straight_macro = tf.summary.scalar('f1_turn_straight_macro', self._scalar)
        self.summary_f1_start_stop_move_micro = tf.summary.scalar('f1_start_stop_move_micro', self._scalar)
        self.summary_f1_start_stop_move_macro = tf.summary.scalar('f1_start_stop_move_macro', self._scalar)
        self.summary_f1_left_right_micro = tf.summary.scalar('f1_left_right_micro', self._scalar)
        self.summary_f1_left_right_macro = tf.summary.scalar('f1_left_right_macro', self._scalar)
        self.summary_brier_score_wait_motion = tf.summary.scalar('brier_wait_motion', self._scalar)
        self.summary_brier_score_turn_straight = tf.summary.scalar('brier_turn_straight', self._scalar)
        self.summary_brier_score_start_stop_move = tf.summary.scalar('brier_start_stop_move', self._scalar)
        self.summary_brier_score_left_right = tf.summary.scalar('brier_left_right', self._scalar)

    ##  Defines the optimizer used for the training
    #   select one of the with self.optimizer_name
    #   ['Adam','RMSProp','GradientDescent','Nesterov','Momentum']
    def define_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.train_wait_motion:
                self.optimizer_wait_motion = self.optimizer_def.minimize(self.loss_wait_motion)
            if self.train_turn_straight:
                self.optimizer_turn_straight = self.optimizer_def.minimize(self.loss_turn_straight)
            if self.train_left_right:
                self.optimizer_left_right = self.optimizer_def.minimize(self.loss_left_right)
            if self.train_start_stop_move:
                self.optimizer_start_stop_move = self.optimizer_def.minimize(self.loss_start_stop_move)
            print('Optimizer ' + self.optimizer_name + ' Set')

    ##  This method is used to build the tensorflow graph
    #   the tensorflow graph is dynamically placed on the most suitable GPU
    def build_graph(self):
        import json
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
                         "_save_False"]
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

        # Create tf session
        # GPU options
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)

        # Create tf session
        if self.optimizer_name == 'Adam':
            self.optimizer_def = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Nadam':
            self.optimizer_def = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'RMSProp':
            self.optimizer_def = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'GradientDescent':
            self.optimizer_def = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Nesterov':
            self.optimizer_def = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                            use_nesterov=True)
        elif self.optimizer_name == 'Momentum':
            self.optimizer_def = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        else:
            self.optimizer_def = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.optimizer_def = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optimizer_def)
        self.sess = tf.Session(config=config)

        used_gpu = '/device:GPU:' + str(self.used_gpu)
        print("Used gpu", used_gpu)
        with tf.device(used_gpu):
            # is_train is necessary for layers that are only active during the training
            self.is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
            # define place holders
            self.create_IO()

            # define run graph
            self.run_graph()

            # Loss function
            self.get_loss()
            self.define_optimizer()

        # Saver
        self.saver = tf.train.Saver(max_to_keep=0)

        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            print([str(i.name) for i in not_initialized_vars])  # only for testing
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        # Initialize Session
        initialize_uninitialized(self.sess)
        self.session_acitve = True

    ## create tensorflow inference graph
    #
    def run_graph(self):
        # create graph based on chosen config
        with tf.name_scope("Classification"):
            if self.use_ms_hk1:
                # use image sequences from camera 1
                with tf.name_scope('conv_net_ms_HK1'):
                    with tf.variable_scope('conv_net_ms'):
                        output = self._conv_net(self.input_ms_hk1)
                        output = flatten(output)
            if self.use_ms_hk2:
                # use image sequences from camera 2
                with tf.name_scope('conv_net_ms_HK2'):
                    with tf.variable_scope('conv_net_ms', reuse=self.use_ms_hk1):
                        conv_net_ms_hk2 = self._conv_net(self.input_ms_hk2)
                        conv_net_ms_hk2 = flatten(conv_net_ms_hk2)
                        if self.use_ms_hk1:
                            output = tf.concat((output, conv_net_ms_hk2), axis=1)
                        else:
                            output = conv_net_ms_hk2
            if self.use_of_hk1:
                # use of sequences from camera 1
                with tf.name_scope('conv_net_of_HK1'):
                    with tf.variable_scope('conv_net_of'):
                        conv_net_of_hk1 = self._conv_net(self.input_of_hk1)
                        conv_net_of_hk1 = flatten(conv_net_of_hk1)
                        if self.use_ms_hk1 or self.use_ms_hk2:
                            output = tf.concat((output, conv_net_of_hk1), axis=1)
                        else:
                            output = conv_net_of_hk1
            if self.use_of_hk2:
                # use of sequences from camera 2
                with tf.name_scope('conv_net_of_HK2'):
                    with tf.variable_scope('conv_net_of', reuse=self.use_of_hk1):
                        conv_net_hk2 = self._conv_net(self.input_of_hk2)
                        conv_net_hk2 = flatten(conv_net_hk2)
                        if self.use_ms_hk1 or self.use_ms_hk2 or self.use_of_hk1:
                            output = tf.concat((output, conv_net_hk2), axis=1)
                        else:
                            output = conv_net_hk2
            if self.use_trajectory:
                # use trajectory
                with tf.name_scope('trajectory_net'):
                    trajectory_net = hidden_architecture(flatten(self.input_trajectory),
                                                         architecture=self.hidden_architecture_trajectory)
                    if self.use_ms_hk1 or self.use_ms_hk2 or self.use_of_hk1 or self.use_of_hk2:
                        output = tf.concat((output, trajectory_net), axis=1)
                    else:
                        output = trajectory_net

            # concatenate outputs from previous graph paths
            with tf.name_scope("Feature_Union"):
                if self.train_wait_motion:
                    output_wait_motion = hidden_architecture(output,
                                                             architecture=self.hidden_architecture_feature_union)
                if self.train_turn_straight:
                    output_turn_straight = hidden_architecture(output,
                                                               architecture=self.hidden_architecture_feature_union)
                if self.train_left_right:
                    output_left_right = hidden_architecture(output, architecture=self.hidden_architecture_feature_union)
                if self.train_start_stop_move:
                    output_start_stop_straight = hidden_architecture(output,
                                                                     architecture=self.hidden_architecture_feature_union)

            # create logits/probabilities
            with tf.name_scope("Linear"):
                if self.train_wait_motion:
                    self.prediction_wait_motion = tf.layers.dense(output_wait_motion, units=2)
                    self.prediction_wait_motion_proba = tf.nn.softmax(self.prediction_wait_motion)
                if self.train_turn_straight:
                    self.prediction_turn_straight = tf.layers.dense(output_turn_straight, units=2)
                    self.prediction_turn_straight_proba = tf.nn.softmax(self.prediction_turn_straight)
                if self.train_left_right:
                    self.prediction_left_right = tf.layers.dense(output_left_right, units=2)
                    self.prediction_left_right_proba = tf.nn.softmax(self.prediction_left_right)
                if self.train_start_stop_move:
                    self.prediction_start_stop_move = tf.layers.dense(output_start_stop_straight, units=3)
                    self.prediction_start_stop_move_proba = tf.nn.softmax(self.prediction_start_stop_move)

    ## create tensorflow loss graph
    #
    def get_loss(self):
        # Cross Entropy Loss
        with tf.name_scope('cross_entropy'):
            if self.train_wait_motion:
                self.loss_wait_motion = tf.losses.softmax_cross_entropy(
                    logits=self.prediction_wait_motion + 0.00001,
                    onehot_labels=self.labels_wait_motion)
            if self.train_turn_straight:
                self.loss_turn_straight = tf.losses.softmax_cross_entropy(
                    logits=self.prediction_turn_straight + 0.00001,
                    onehot_labels=self.labels_turn_straight)
            if self.train_left_right:
                self.loss_left_right = tf.losses.softmax_cross_entropy(
                    logits=self.prediction_left_right + 0.00001,
                    onehot_labels=self.labels_left_right)
            if self.train_start_stop_move:
                self.loss_start_stop_move = tf.losses.softmax_cross_entropy(
                    logits=self.prediction_start_stop_move + 0.00001,
                    onehot_labels=self.labels_start_stop_move)

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
    def fit(self, X, y=None, train=None, validation=None, scoring=False, steps=1, print_steps=None,
            score_steps=None,
            cyclic_steps=None, **kwargs):

        # Build Tensorflow Graph
        if not self.resume_fit:
            self.build_graph()

        if self.train_wait_motion:
            self.train_buffer_wait_motion = Queue(maxsize=self.buffer_size)
            self.valid_buffer_wait_motion = Queue(maxsize=self.buffer_size)
        if self.train_turn_straight:
            self.train_buffer_turn_straight = Queue(maxsize=self.buffer_size)
            self.valid_buffer_turn_straight = Queue(maxsize=self.buffer_size)
        if self.train_left_right:
            self.train_buffer_left_right = Queue(maxsize=self.buffer_size)
            self.valid_buffer_left_right = Queue(maxsize=self.buffer_size)
        if self.train_start_stop_move:
            self.train_buffer_start_stop_move = Queue(maxsize=self.buffer_size)
            self.valid_buffer_start_stop_move = Queue(maxsize=self.buffer_size)

        self.buffering = False

        # check if scoring is possible
        if not isinstance(validation, type(None)):
            valid_data_exists = True
        else:
            valid_data_exists = False

        if scoring and not valid_data_exists:
            print('fold_' + str(self.fold), "WARNING: Cant score without validation data!")
            scoring = False

        # split into train and test data
        if valid_data_exists and type(X) == dict and y is not None:
            X_train = X['X_train']
            y_train = y[train]
            X_valid = X['X_validation']
            y_valid = y[validation]
        elif valid_data_exists and y is not None:
            X_train = X[train]
            y_train = y[train]
            X_valid = X[validation]
            y_valid = y[validation]
        elif valid_data_exists and y is None:
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

        if print_steps:
            print(
                'fold_' + str(self.fold), "Prints current intermediate train loss every %s'th epoch" % print_steps)
        else:
            print_steps = steps * 3

        if score_steps and scoring == True:
            print('fold_' + str(self.fold), "Score the model on validation data every %s'th epoch" % score_steps)
            # check if user defined scoring method is found
            if 'get_score' not in self.overwritten_methods:
                print(
                    'fold_' + str(self.fold),
                    "WARNING: No user defined scoring method is found, using 1/SSE as score!")

        if score_steps and scoring == False and valid_data_exists == True:
            print('fold_' + str(self.fold), "Computes loss on validation data every %s'th epoch" % score_steps)
        if score_steps == None:
            score_steps = steps * 3

        if self.save == True:
            print('fold_' + str(self.fold), "Save the model every %s'th epoch" % score_steps)

        if not os.path.exists(self.param_path):
            os.makedirs(self.param_path)

        self.buffering = True
        # start buffering data
        if self.train_wait_motion:
            self.buffer_train_data_wait_motion = [
                Process(target=self.load_buffer,
                        args=(
                            X_train, y_train, self.train_buffer_wait_motion, 'wait_motion', self.balance_samples))
                for
                _ in
                range(1)]
            for thread in self.buffer_train_data_wait_motion:
                thread.start()
            self.buffer_valid_data_wait_motion = [
                Process(target=self.load_buffer,
                        args=(
                            X_valid, y_valid, self.valid_buffer_wait_motion, 'wait_motion', self.balance_samples))
                for
                _
                in
                range(1)]
            for thread in self.buffer_valid_data_wait_motion:
                thread.start()
        if self.train_turn_straight:
            buffer_train_data_turn_straight = [
                Process(target=self.load_buffer,
                        args=(
                            X_train, y_train, self.train_buffer_turn_straight, 'turn_straight',
                            self.balance_samples))
                for _ in
                range(1)]
            for thread in buffer_train_data_turn_straight:
                thread.start()
            buffer_valid_data_turn_straight = [
                Process(target=self.load_buffer,
                        args=(
                            X_valid, y_valid, self.valid_buffer_turn_straight, 'turn_straight',
                            self.balance_samples))
                for
                _ in
                range(1)]
            for thread in buffer_valid_data_turn_straight:
                thread.start()
        if self.train_left_right:
            buffer_train_data_left_right = [
                Process(target=self.load_buffer,
                        args=(
                            X_train, y_train, self.train_buffer_left_right, 'left_right',
                            self.balance_samples))
                for _
                in
                range(1)]
            for thread in buffer_train_data_left_right:
                thread.start()
            buffer_valid_data_left_right = [
                Process(target=self.load_buffer,
                        args=(
                            X_valid, y_valid, self.valid_buffer_left_right, 'left_right',
                            self.balance_samples))
                for
                _
                in
                range(1)]
            for thread in buffer_valid_data_left_right:
                thread.start()
        if self.train_start_stop_move:
            buffer_train_data_start_stop_move = [
                Process(target=self.load_buffer,
                        args=(X_train, y_train, self.train_buffer_start_stop_move, 'start_stop_move',
                              self.balance_samples)) for
                _ in
                range(1)]
            for thread in buffer_train_data_start_stop_move:
                thread.start()
            buffer_valid_data_start_stop_move = [
                Process(target=self.load_buffer,
                        args=(X_valid, y_valid, self.valid_buffer_start_stop_move, 'start_stop_move',
                              self.balance_samples))
                for
                _ in
                range(1)]
            for thread in buffer_valid_data_start_stop_move:
                thread.start()

        # load the current hyperparams
        hyperparams = self.__dict__
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
        train_loss_ = tf.placeholder(tf.float32)
        train_loss_wait_motion_ = tf.placeholder(tf.float32)
        train_loss_turn_straight_ = tf.placeholder(tf.float32)
        train_loss_left_right_ = tf.placeholder(tf.float32)
        train_loss_start_stop_move_ = tf.placeholder(tf.float32)
        valid_loss_ = tf.placeholder(tf.float32)
        valid_loss_wait_motion_ = tf.placeholder(tf.float32)
        valid_loss_turn_straight_ = tf.placeholder(tf.float32)
        valid_loss_left_right_ = tf.placeholder(tf.float32)
        valid_loss_start_stop_move_ = tf.placeholder(tf.float32)
        score_ = tf.placeholder(tf.float32)

        # Tensor board file writer
        self.file_writer = tf.summary.FileWriter(self.param_path + '/train',
                                                 self.sess.graph)

        train_loss_summary_ = tf.summary.scalar('train_loss', train_loss_)
        train_loss_summary_wait_motion_ = tf.summary.scalar('train_loss_wait_motion', train_loss_wait_motion_)
        train_loss_summary_turn_straight_ = tf.summary.scalar('train_loss_turn_straight', train_loss_turn_straight_)
        train_loss_summary_left_right_ = tf.summary.scalar('train_loss_left_right', train_loss_left_right_)
        train_loss_summary_start_stop_move_ = tf.summary.scalar('train_loss_start_stop_move',
                                                                train_loss_start_stop_move_)
        valid_loss_summary_ = tf.summary.scalar('valid_loss', valid_loss_)
        valid_loss_summary_wait_motion_ = tf.summary.scalar('valid_loss_wait_motion', valid_loss_wait_motion_)
        valid_loss_summary_turn_straight_ = tf.summary.scalar('valid_loss_turn_straight', valid_loss_turn_straight_)
        valid_loss_summary_left_right_ = tf.summary.scalar('valid_loss_left_right', valid_loss_left_right_)
        valid_loss_summary_start_stop_move_ = tf.summary.scalar('valid_loss_start_stop_move',
                                                                valid_loss_start_stop_move_)
        valid_score_summary_ = tf.summary.scalar('valid_score', score_)

        # Train the model

        # fit loop
        epoch = self.start_epoch
        train_range = range(self.start_epoch, steps + 1)
        for epoch in train_range:
            gc.collect()

            X_train_wait_motion_unzipped = []
            y_train_wait_motion_unzipped = []
            if self.train_wait_motion:
                batch = self.train_buffer_wait_motion.get()
                for sample in batch:
                    X_train_wait_motion_unzipped.append(sample['X'])
                    y_train_wait_motion_unzipped.append(sample['y'])
            X_train_turn_straight_unzipped = []
            y_train_turn_straight_unzipped = []
            if self.train_turn_straight:
                batch = self.train_buffer_turn_straight.get()
                for sample in batch:
                    X_train_turn_straight_unzipped.append(sample['X'])
                    y_train_turn_straight_unzipped.append(sample['y'])
            X_train_left_right_unzipped = []
            y_train_left_right_unzipped = []
            if self.train_left_right:
                batch = self.train_buffer_left_right.get()
                for sample in batch:
                    X_train_left_right_unzipped.append(sample['X'])
                    y_train_left_right_unzipped.append(sample['y'])
            X_train_start_stop_move_unzipped = []
            y_train_start_stop_move_unzipped = []
            if self.train_start_stop_move:
                batch = self.train_buffer_start_stop_move.get()
                for sample in batch:
                    X_train_start_stop_move_unzipped.append(sample['X'])
                    y_train_start_stop_move_unzipped.append(sample['y'])

            X_train_dict = {'wait_motion': X_train_wait_motion_unzipped,
                            'turn_straight': X_train_turn_straight_unzipped,
                            'left_right': X_train_left_right_unzipped,
                            'start_stop_move': X_train_start_stop_move_unzipped}
            y_train_dict = {'wait_motion': y_train_wait_motion_unzipped,
                            'turn_straight': y_train_turn_straight_unzipped,
                            'left_right': y_train_left_right_unzipped,
                            'start_stop_move': y_train_start_stop_move_unzipped}

            # get loss
            curr_loss, \
            cur_loss_turn_straight, \
            cur_loss_left_right, \
            cur_loss_wait_motion, \
            cur_loss_start_stop_move = self.fit_epoch(X_train_dict, y_train_dict, **kwargs)

            if self.train_wait_motion:
                train_loss_summary_wait_motion = self.sess.run(train_loss_summary_wait_motion_,
                                                               feed_dict={
                                                                   train_loss_wait_motion_: cur_loss_wait_motion})
                self.file_writer.add_summary(train_loss_summary_wait_motion, global_step=epoch)
            if self.train_turn_straight:
                train_loss_summary_turn_straight = self.sess.run(train_loss_summary_turn_straight_,
                                                                 feed_dict={
                                                                     train_loss_turn_straight_: cur_loss_turn_straight})
                self.file_writer.add_summary(train_loss_summary_turn_straight, global_step=epoch)
            if self.train_left_right:
                train_loss_summary_left_right = self.sess.run(train_loss_summary_left_right_,
                                                              feed_dict={
                                                                  train_loss_left_right_: cur_loss_left_right})
                self.file_writer.add_summary(train_loss_summary_left_right, global_step=epoch)
            if self.train_start_stop_move:
                train_loss_summary_start_stop_move = self.sess.run(train_loss_summary_start_stop_move_,
                                                                   feed_dict={
                                                                       train_loss_start_stop_move_:
                                                                           cur_loss_start_stop_move})
                self.file_writer.add_summary(train_loss_summary_start_stop_move, global_step=epoch)

            train_loss_summary = self.sess.run(train_loss_summary_,
                                               feed_dict={
                                                   train_loss_: curr_loss})
            self.file_writer.add_summary(train_loss_summary, global_step=epoch)

            if epoch % print_steps == 0:
                print(self.fold, "step:", epoch, "loss:", curr_loss,
                      "loss_wait_motion:", cur_loss_wait_motion,
                      "loss_turn_straight:", cur_loss_turn_straight,
                      "loss_left_right:", cur_loss_left_right,
                      "loss_start_stop_move:", cur_loss_start_stop_move,
                      self.name)
            # if the current epoch is a scoring epoch or the last epoch, do the scoring
            if (epoch % score_steps == 0) or epoch == steps or steps == 1:
                if valid_data_exists:
                    valid_score = '--'
                    X_valid_wait_motion_unzipped = []
                    y_valid_wait_motion_unzipped = []
                    if self.train_wait_motion:
                        for _ in range(n_validation_steps):
                            batch = self.valid_buffer_wait_motion.get()
                            for sample in batch:
                                X_valid_wait_motion_unzipped.append(sample['X'])
                                y_valid_wait_motion_unzipped.append(sample['y'])
                    X_valid_turn_straight_unzipped = []
                    y_valid_turn_straight_unzipped = []
                    if self.train_turn_straight:
                        for _ in range(n_validation_steps):
                            batch = self.valid_buffer_turn_straight.get()
                            for sample in batch:
                                X_valid_turn_straight_unzipped.append(sample['X'])
                                y_valid_turn_straight_unzipped.append(sample['y'])
                    X_valid_left_right_unzipped = []
                    y_valid_left_right_unzipped = []
                    if self.train_left_right:
                        for _ in range(n_validation_steps):
                            batch = self.valid_buffer_left_right.get()
                            for sample in batch:
                                X_valid_left_right_unzipped.append(sample['X'])
                                y_valid_left_right_unzipped.append(sample['y'])
                    X_valid_start_stop_move_unzipped = []
                    y_valid_start_stop_move_unzipped = []
                    if self.train_start_stop_move:
                        for _ in range(n_validation_steps):
                            batch = self.valid_buffer_start_stop_move.get()
                            for sample in batch:
                                X_valid_start_stop_move_unzipped.append(sample['X'])
                                y_valid_start_stop_move_unzipped.append(sample['y'])

                    X_valid_dict = {'wait_motion': X_valid_wait_motion_unzipped,
                                    'turn_straight': X_valid_turn_straight_unzipped,
                                    'left_right': X_valid_left_right_unzipped,
                                    'start_stop_move': X_valid_start_stop_move_unzipped}
                    y_valid_dict = {'wait_motion': y_valid_wait_motion_unzipped,
                                    'turn_straight': y_valid_turn_straight_unzipped,
                                    'left_right': y_valid_left_right_unzipped,
                                    'start_stop_move': y_valid_start_stop_move_unzipped}

                    if scoring:
                        # build and save valid score
                        valid_score = self.score(X_valid_dict, y_valid_dict, step=epoch, **kwargs)
                        score_summary_out = self.sess.run(valid_score_summary_,
                                                          feed_dict={score_: valid_score})

                        self.file_writer.add_summary(score_summary_out, global_step=epoch)

                    curr_loss, \
                    cur_loss_wait_motion, \
                    cur_loss_turn_straight, \
                    cur_loss_left_right, \
                    cur_loss_start_stop_move = self.calculate_loss(X_valid_dict, y_valid_dict, **kwargs)

                    if self.train_wait_motion:
                        valid_loss_summary_wait_motion = self.sess.run(valid_loss_summary_wait_motion_,
                                                                       feed_dict={
                                                                           valid_loss_wait_motion_: cur_loss_wait_motion})
                        self.file_writer.add_summary(valid_loss_summary_wait_motion, global_step=epoch)
                    if self.train_turn_straight:
                        valid_loss_summary_turn_straight = self.sess.run(valid_loss_summary_turn_straight_,
                                                                         feed_dict={
                                                                             valid_loss_turn_straight_: cur_loss_turn_straight})
                        self.file_writer.add_summary(valid_loss_summary_turn_straight, global_step=epoch)
                    if self.train_left_right:
                        valid_loss_summary_left_right = self.sess.run(valid_loss_summary_left_right_,
                                                                      feed_dict={
                                                                          valid_loss_left_right_: cur_loss_left_right})
                        self.file_writer.add_summary(valid_loss_summary_left_right, global_step=epoch)
                    if self.train_start_stop_move:
                        valid_loss_summary_start_stop_move = self.sess.run(valid_loss_summary_start_stop_move_,
                                                                           feed_dict={
                                                                               valid_loss_start_stop_move_: cur_loss_start_stop_move})
                        self.file_writer.add_summary(valid_loss_summary_start_stop_move, global_step=epoch)

                    valid_loss_summary = self.sess.run(valid_loss_summary_,
                                                       feed_dict={
                                                           valid_loss_: curr_loss})
                    self.file_writer.add_summary(valid_loss_summary, global_step=epoch)

                    print(self.fold, "validation: step:", epoch,
                          "loss:", curr_loss,
                          "loss_wait_motion:", cur_loss_wait_motion,
                          "loss_turn_straight:", cur_loss_turn_straight,
                          "loss_left_right:", cur_loss_left_right,
                          "loss_start_stop_move:", cur_loss_start_stop_move,
                          "score:", valid_score, self.name)

                    self.save_backup(epoch, valid_score)
                else:
                    self.save_backup(epoch)

                if self.save:
                    # save the weights and biases of the model (tf variables) in a checkpoint-file
                    self.saver.save(self.sess,
                                    os.path.join(self.path, hashlib.sha256(self.name.encode('utf-8')).hexdigest(),
                                                 str(epoch),
                                                 'model.ckpt'))
                    # save the current epoch in a pickle-file
                    with open(os.path.join(self.param_path, 'last_epoch.pkl'), 'wb') as epoch_file:
                        # dump the file
                        pickle.dump(str(epoch), epoch_file)
                        # check that the file is properly synced on disk
                        os.fsync(epoch_file)

        # save the current epoch in a pickle-file
        with open(os.path.join(self.param_path, 'last_epoch.pkl'), 'wb') as epoch_file:
            pickle.dump(str(epoch), epoch_file)
            # check that the file is properly synced on disk
            os.fsync(epoch_file)

        # stop buffering threads
        # stop buffering data
        print('terminating processes')
        self.buffering = False
        if self.train_wait_motion:
            for thread in self.buffer_train_data_wait_motion:
                thread.terminate()
            for thread in self.buffer_valid_data_wait_motion:
                thread.terminate()
        if self.train_turn_straight:
            for thread in buffer_train_data_turn_straight:
                thread.terminate()
            for thread in buffer_valid_data_turn_straight:
                thread.terminate()
        if self.train_left_right:
            for thread in buffer_train_data_left_right:
                thread.terminate()
            for thread in buffer_valid_data_left_right:
                thread.terminate()
        if self.train_start_stop_move:
            for thread in buffer_train_data_start_stop_move:
                thread.terminate()
            for thread in buffer_valid_data_start_stop_move:
                thread.terminate()

        print('processes terminated')

    # fit one epoch
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    def fit_epoch(self, X, y, **kwargs):
        X_wait_motion = X['wait_motion']
        y_wait_motion = y['wait_motion']
        X_turn_straight = X['turn_straight']
        y_turn_straight = y['turn_straight']
        X_left_right = X['left_right']
        y_left_right = y['left_right']
        X_start_stop_move = X['start_stop_move']
        y_start_stop_move = y['start_stop_move']

        # fit turn straight and left right state machines
        ms_hk1 = []
        ms_hk2 = []
        of_hk1 = []
        of_hk2 = []
        trajectories = []
        for dic in X_wait_motion:
            if self.use_ms_hk1:
                ms_hk1.append(dic['ms_hk1'])
            if self.use_ms_hk2:
                ms_hk2.append(dic['ms_hk2'])
            if self.use_of_hk1:
                of_hk1.append(dic['of_hk1'])
            if self.use_of_hk2:
                of_hk2.append(dic['of_hk2'])
            if self.use_trajectory:
                trajectories.append(dic['trajectory'])

        labels = []
        for dic in y_wait_motion:
            labels.append(dic['labels_wait_motion'])

        # check if list contains items
        if len(X_wait_motion) > 0:
            feed_dict = {self.labels_wait_motion: labels,
                         self.training_flag: True}
            if self.use_ms_hk1:
                feed_dict.update({self.input_ms_hk1: np.swapaxes(np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_ms_hk2:
                feed_dict.update({self.input_ms_hk2: np.swapaxes(np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_of_hk1:
                feed_dict.update({self.input_of_hk1: of_hk1})
            if self.use_of_hk2:
                feed_dict.update({self.input_of_hk2: of_hk2})
            if self.use_trajectory:
                feed_dict.update({self.input_trajectory: trajectories})

            _, cur_loss_wait_motion = self.sess.run([self.optimizer_wait_motion, self.loss_wait_motion],
                                                    feed_dict=feed_dict)
        else:
            cur_loss_wait_motion = 0

        # fit left straight turn state machine
        ms_hk1 = []
        ms_hk2 = []
        of_hk1 = []
        of_hk2 = []
        trajectories = []
        for dic in X_turn_straight:
            if self.use_ms_hk1:
                ms_hk1.append(dic['ms_hk1'])
            if self.use_ms_hk2:
                ms_hk2.append(dic['ms_hk2'])
            if self.use_of_hk1:
                of_hk1.append(dic['of_hk1'])
            if self.use_of_hk2:
                of_hk2.append(dic['of_hk2'])
            if self.use_trajectory:
                trajectories.append(dic['trajectory'])

        labels = []
        for dic in y_turn_straight:
            labels.append(dic['labels_turn_straight'])

        # check if list contains items
        if len(X_turn_straight) > 0:
            feed_dict = {self.labels_turn_straight: labels,
                         self.training_flag: True}
            if self.use_ms_hk1:
                feed_dict.update({self.input_ms_hk1: np.swapaxes(np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_ms_hk2:
                feed_dict.update({self.input_ms_hk2: np.swapaxes(np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_of_hk1:
                feed_dict.update({self.input_of_hk1: of_hk1})
            if self.use_of_hk2:
                feed_dict.update({self.input_of_hk2: of_hk2})
            if self.use_trajectory:
                feed_dict.update({self.input_trajectory: trajectories})
            _, cur_loss_turn_straight = self.sess.run([self.optimizer_turn_straight, self.loss_turn_straight],
                                                      feed_dict=feed_dict)
        else:
            cur_loss_turn_straight = 0

        # fit left right state machine
        ms_hk1 = []
        ms_hk2 = []
        of_hk1 = []
        of_hk2 = []
        trajectories = []
        for dic in X_left_right:
            if self.use_ms_hk1:
                ms_hk1.append(dic['ms_hk1'])
            if self.use_ms_hk2:
                ms_hk2.append(dic['ms_hk2'])
            if self.use_of_hk1:
                of_hk1.append(dic['of_hk1'])
            if self.use_of_hk2:
                of_hk2.append(dic['of_hk2'])
            if self.use_trajectory:
                trajectories.append(dic['trajectory'])

        labels = []
        for dic in y_left_right:
            labels.append(dic['labels_left_right'])

        # check if list contains items
        if len(X_left_right) > 0:
            feed_dict = {self.labels_left_right: labels,
                         self.training_flag: True}
            if self.use_ms_hk1:
                feed_dict.update({self.input_ms_hk1: np.swapaxes(np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_ms_hk2:
                feed_dict.update({self.input_ms_hk2: np.swapaxes(np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_of_hk1:
                feed_dict.update({self.input_of_hk1: of_hk1})
            if self.use_of_hk2:
                feed_dict.update({self.input_of_hk2: of_hk2})
            if self.use_trajectory:
                feed_dict.update({self.input_trajectory: trajectories})

            _, cur_loss_left_right = self.sess.run([self.optimizer_left_right, self.loss_left_right],
                                                   feed_dict=feed_dict)
        else:
            cur_loss_left_right = 0

        # fit start stop move state machine
        ms_hk1 = []
        ms_hk2 = []
        of_hk1 = []
        of_hk2 = []
        trajectories = []
        for dic in X_start_stop_move:
            if self.use_ms_hk1:
                ms_hk1.append(dic['ms_hk1'])
            if self.use_ms_hk2:
                ms_hk2.append(dic['ms_hk2'])
            if self.use_of_hk1:
                of_hk1.append(dic['of_hk1'])
            if self.use_of_hk2:
                of_hk2.append(dic['of_hk2'])
            if self.use_trajectory:
                trajectories.append(dic['trajectory'])

        labels = []
        for dic in y_start_stop_move:
            labels.append(dic['labels_start_stop_move'])

        # check if list contains items
        if len(X_start_stop_move) > 0:
            feed_dict = {self.labels_start_stop_move: labels,
                         self.training_flag: True}
            if self.use_ms_hk1:
                feed_dict.update({self.input_ms_hk1: np.swapaxes(np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_ms_hk2:
                feed_dict.update({self.input_ms_hk2: np.swapaxes(np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2),
                                                                 2, 3) / 255.0})
            if self.use_of_hk1:
                feed_dict.update({self.input_of_hk1: of_hk1})
            if self.use_of_hk2:
                feed_dict.update({self.input_of_hk2: of_hk2})
            if self.use_trajectory:
                feed_dict.update({self.input_trajectory: trajectories})

            _, cur_loss_start_stop_move = self.sess.run([self.optimizer_start_stop_move, self.loss_start_stop_move],
                                                        feed_dict=feed_dict)
        else:
            cur_loss_start_stop_move = 0

        return cur_loss_turn_straight + cur_loss_left_right + cur_loss_wait_motion + cur_loss_start_stop_move, \
               cur_loss_turn_straight, cur_loss_left_right, cur_loss_wait_motion, cur_loss_start_stop_move

    ##  calculates loss (optional in bathes if self.batch_size is not None)
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    the loss for the data
    def calculate_loss(self, X, y, **kwargs):

        X_wait_motion = X['wait_motion']
        y_wait_motion = y['wait_motion']
        X_turn_straight = X['turn_straight']
        y_turn_straight = y['turn_straight']
        X_left_right = X['left_right']
        y_left_right = y['left_right']
        X_start_stop_move = X['start_stop_move']
        y_start_stop_move = y['start_stop_move']

        # calculate loss of lateral state machine
        if len(X_wait_motion) > 0:
            loss_wait_motion = []
            for i in range(n_validation_steps):

                batch_X = X_wait_motion[i * self.batch_size:i * self.batch_size + self.batch_size]
                batch_y = y_wait_motion[i * self.batch_size:i * self.batch_size + self.batch_size]

                ms_hk1 = []
                ms_hk2 = []
                of_hk1 = []
                of_hk2 = []
                trajectories = []
                for dic in batch_X:
                    if self.use_ms_hk1:
                        ms_hk1.append(dic['ms_hk1'])
                    if self.use_ms_hk2:
                        ms_hk2.append(dic['ms_hk2'])
                    if self.use_of_hk1:
                        of_hk1.append(dic['of_hk1'])
                    if self.use_of_hk2:
                        of_hk2.append(dic['of_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                labels = []
                for dic in batch_y:
                    labels.append(dic['labels_wait_motion'])

                if len(batch_X) > 0:
                    feed_dict = {self.labels_wait_motion: labels,
                                 self.training_flag: False}
                    if self.use_ms_hk1:
                        feed_dict.update({self.input_ms_hk1: np.swapaxes(
                            np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_ms_hk2:
                        feed_dict.update({self.input_ms_hk2: np.swapaxes(
                            np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_of_hk1:
                        feed_dict.update({self.input_of_hk1: of_hk1})
                    if self.use_of_hk2:
                        feed_dict.update({self.input_of_hk2: of_hk2})
                    if self.use_trajectory:
                        feed_dict.update({self.input_trajectory: trajectories})

                    batch_loss_wait_motion = self.sess.run(self.loss_wait_motion, feed_dict=feed_dict)

                    loss_wait_motion.append(batch_loss_wait_motion)

            cur_loss_wait_motion = np.mean(np.array(loss_wait_motion))
        else:
            cur_loss_wait_motion = 0

        # calculate loss of lateral state machine
        if len(X_turn_straight) > 0:
            loss_turn_straight = []
            for i in range(n_validation_steps):

                batch_X = X_turn_straight[i * self.batch_size:i * self.batch_size + self.batch_size]
                batch_y = y_turn_straight[i * self.batch_size:i * self.batch_size + self.batch_size]

                ms_hk1 = []
                ms_hk2 = []
                of_hk1 = []
                of_hk2 = []
                trajectories = []
                for dic in batch_X:
                    if self.use_ms_hk1:
                        ms_hk1.append(dic['ms_hk1'])
                    if self.use_ms_hk2:
                        ms_hk2.append(dic['ms_hk2'])
                    if self.use_of_hk1:
                        of_hk1.append(dic['of_hk1'])
                    if self.use_of_hk2:
                        of_hk2.append(dic['of_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                labels = []
                for dic in batch_y:
                    labels.append(dic['labels_turn_straight'])

                if len(batch_X) > 0:
                    feed_dict = {self.labels_turn_straight: labels,
                                 self.training_flag: False}
                    if self.use_ms_hk1:
                        feed_dict.update({self.input_ms_hk1: np.swapaxes(
                            np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_ms_hk2:
                        feed_dict.update({self.input_ms_hk2: np.swapaxes(
                            np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_of_hk1:
                        feed_dict.update({self.input_of_hk1: of_hk1})
                    if self.use_of_hk2:
                        feed_dict.update({self.input_of_hk2: of_hk2})
                    if self.use_trajectory:
                        feed_dict.update({self.input_trajectory: trajectories})

                    batch_loss_turn_straight = self.sess.run(self.loss_turn_straight, feed_dict=feed_dict)

                    loss_turn_straight.append(batch_loss_turn_straight)

            cur_loss_turn_straight = np.mean(np.array(loss_turn_straight))
        else:
            cur_loss_turn_straight = 0

        # calculate loss of lateral state machine
        if len(X_left_right):
            loss_left_right = []
            for i in range(n_validation_steps):

                batch_X = X_left_right[i * self.batch_size:i * self.batch_size + self.batch_size]
                batch_y = y_left_right[i * self.batch_size:i * self.batch_size + self.batch_size]

                ms_hk1 = []
                ms_hk2 = []
                of_hk1 = []
                of_hk2 = []
                trajectories = []
                for dic in batch_X:
                    if self.use_ms_hk1:
                        ms_hk1.append(dic['ms_hk1'])
                    if self.use_ms_hk2:
                        ms_hk2.append(dic['ms_hk2'])
                    if self.use_of_hk1:
                        of_hk1.append(dic['of_hk1'])
                    if self.use_of_hk2:
                        of_hk2.append(dic['of_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                labels = []
                for dic in batch_y:
                    labels.append(dic['labels_left_right'])

                if len(batch_X) > 0:
                    feed_dict = {self.labels_left_right: labels,
                                 self.training_flag: False}
                    if self.use_ms_hk1:
                        feed_dict.update({self.input_ms_hk1: np.swapaxes(
                            np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_ms_hk2:
                        feed_dict.update({self.input_ms_hk2: np.swapaxes(
                            np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_of_hk1:
                        feed_dict.update({self.input_of_hk1: of_hk1})
                    if self.use_of_hk2:
                        feed_dict.update({self.input_of_hk2: of_hk2})
                    if self.use_trajectory:
                        feed_dict.update({self.input_trajectory: trajectories})

                    batch_loss_left_right = self.sess.run(self.loss_left_right, feed_dict=feed_dict)

                    loss_left_right.append(batch_loss_left_right)

            cur_loss_left_right = np.mean(np.array(loss_left_right))
        else:
            cur_loss_left_right = 0

        # calculate loss of lateral state machine
        if len(X_start_stop_move) > 0:
            loss_start_stop_move = []
            for i in range(n_validation_steps):

                batch_X = X_start_stop_move[i * self.batch_size:i * self.batch_size + self.batch_size]
                batch_y = y_start_stop_move[i * self.batch_size:i * self.batch_size + self.batch_size]

                ms_hk1 = []
                ms_hk2 = []
                of_hk1 = []
                of_hk2 = []
                trajectories = []
                for dic in batch_X:
                    if self.use_ms_hk1:
                        ms_hk1.append(dic['ms_hk1'])
                    if self.use_ms_hk2:
                        ms_hk2.append(dic['ms_hk2'])
                    if self.use_of_hk1:
                        of_hk1.append(dic['of_hk1'])
                    if self.use_of_hk2:
                        of_hk2.append(dic['of_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                labels = []
                for dic in batch_y:
                    labels.append(dic['labels_start_stop_move'])

                if len(batch_X) > 0:
                    feed_dict = {self.labels_start_stop_move: labels,
                                 self.training_flag: False}
                    if self.use_ms_hk1:
                        feed_dict.update({self.input_ms_hk1: np.swapaxes(
                            np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_ms_hk2:
                        feed_dict.update({self.input_ms_hk2: np.swapaxes(
                            np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                    if self.use_of_hk1:
                        feed_dict.update({self.input_of_hk1: of_hk1})
                    if self.use_of_hk2:
                        feed_dict.update({self.input_of_hk2: of_hk2})
                    if self.use_trajectory:
                        feed_dict.update({self.input_trajectory: trajectories})

                    batch_loss_start_stop_move = self.sess.run(self.loss_start_stop_move, feed_dict=feed_dict)

                    loss_start_stop_move.append(batch_loss_start_stop_move)

            cur_loss_start_stop_move = np.mean(np.array(loss_start_stop_move))
        else:
            cur_loss_start_stop_move = 0

        return cur_loss_wait_motion + cur_loss_turn_straight + cur_loss_left_right + cur_loss_start_stop_move, \
               cur_loss_wait_motion, cur_loss_turn_straight, cur_loss_left_right, cur_loss_start_stop_move

    ## computes score (optional in batches if self.batch_size is not None)
    #   overload _get_score for your needs
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param **kwargs:            variable number of keyword arguments
    #   @return:                    score
    def score(self, X, y, step=None, **kwargs):
        score = self.get_score(X, y, **kwargs)
        self.cyclic_(X, y, step=step)
        return score

    ## computes score used to pick best estimator
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param step:                current training step
    #   @return:                    score
    def get_score(self, X, y=None, step=None):
        loss, _, _, _, _ = self.calculate_loss(X, y=y)
        return 1 / loss

    ## creates evaluation plots during vlaidation steps
    #   @param X:                   input data
    #   @param y:                   ground truth data
    #   @param step:                current training step
    #   @param **kwargs:            variable number of keyword arguments
    def cyclic_(self, X, y, step=None, **kwargs):
        if self.mutex is not None:
            self.mutex.acquire()

        X_wait_motion = X['wait_motion']
        y_wait_motion = y['wait_motion']
        X_turn_straight = X['turn_straight']
        y_turn_straight = y['turn_straight']
        X_left_right = X['left_right']
        y_left_right = y['left_right']
        X_start_stop_move = X['start_stop_move']
        y_start_stop_move = y['start_stop_move']

        # calculate scores
        def caclulate_scores(X, y, state_machine='wait_motion'):
            state_machines = ['wait_motion', 'turn_straight', 'left_right', 'start_stop_move']
            prediction = []
            labels = []
            for i in range(n_validation_steps):
                # get random samples from training data
                batch_X = X[i * self.batch_size:i * self.batch_size + self.batch_size]
                batch_y = y[i * self.batch_size:i * self.batch_size + self.batch_size]

                # calculate F1 scores and confusion matrices for lon and lat predictions
                pred_out = self.predict(batch_X)
                pred = pred_out[state_machines.index(state_machine)]
                prediction.extend(pred)
                for dict in batch_y:
                    labels.append(dict['labels_' + state_machine])

            pred_class = np.argmax(prediction, axis=1)
            true_class = np.argmax(labels, axis=1)
            f1_micro = f1_score(true_class, pred_class, average='micro')
            f1_macro = f1_score(true_class, pred_class, average='macro')
            brier_score = np.mean(
                [brier_score_loss(y_true=labels[i], y_prob=prediction[i]) for i in range(len(prediction))])

            return pred_class, true_class, f1_micro, f1_macro, np.array(labels), np.array(prediction), brier_score

        # wait_motion
        if len(X_wait_motion) > 0:
            # calculate f scores of all state machines
            pred_class_wait_motion, \
            true_class_wait_motion, \
            f1_wait_motion_macro, \
            f1_wait_motion_micro, \
            true_one_hot_wait_motion, \
            pred_one_hot_wait_motion, \
            brier_wait_motion = caclulate_scores(X_wait_motion, y_wait_motion, state_machine='wait_motion')

            # create qq plots
            qq_path = os.path.join(self.param_path, 'qq', str(step))
            if not os.path.exists(qq_path):
                os.makedirs(qq_path)
            true_wait = true_one_hot_wait_motion[:, 0]
            pred_wait = pred_one_hot_wait_motion[:, 0]
            create_qq_plot(y_true=true_wait, y_pred=pred_wait, title='wait')
            plt.savefig(os.path.join(qq_path, 'wait.pdf'))
            plt.close()
            true_motion = true_one_hot_wait_motion[:, 1]
            pred_motion = pred_one_hot_wait_motion[:, 1]
            create_qq_plot(y_true=true_motion, y_pred=pred_motion, title='motion')
            plt.savefig(os.path.join(qq_path, 'motion.pdf'))
            plt.close()

            # write summaries to tensorboard
            # write summaries
            summary_f1_wait_motion_micro = self.sess.run(self.summary_f1_wait_motion_micro,
                                                         feed_dict={self._scalar: f1_wait_motion_micro})
            self.file_writer.add_summary(summary_f1_wait_motion_micro, global_step=step)
            summary_f1_wait_motion_macro = self.sess.run(self.summary_f1_wait_motion_macro,
                                                         feed_dict={self._scalar: f1_wait_motion_macro})
            self.file_writer.add_summary(summary_f1_wait_motion_macro, global_step=step)
            summary_brier_wait_motion = self.sess.run(self.summary_brier_score_wait_motion,
                                                      feed_dict={self._scalar: brier_wait_motion})
            self.file_writer.add_summary(summary_brier_wait_motion, global_step=step)
            # plot confusion matrices
            conf_matrix_path = os.path.join(self.param_path, 'conufsion_matrices')
            if not os.path.exists(conf_matrix_path):
                os.makedirs(conf_matrix_path)
            conf_matrix_path_step = os.path.join(conf_matrix_path, str(step))
            if not os.path.exists(conf_matrix_path_step):
                os.makedirs(conf_matrix_path_step)
            plot_confusion_matrix(y_true=true_class_wait_motion, y_pred=pred_class_wait_motion,
                                  classes=['wait', 'motion'],
                                  title='Wait Motion', normalize=False)
            plt.savefig(os.path.join(conf_matrix_path_step, 'WaitMotion.pdf'))
            plt.close()
            plot_confusion_matrix(y_true=true_class_wait_motion, y_pred=pred_class_wait_motion,
                                  classes=['wait', 'motion'],
                                  title='Wait Motion', normalize=True)
            plt.savefig(os.path.join(conf_matrix_path_step, 'WaitMotionNorm.pdf'))
            plt.close()

        # turn_straight
        if len(X_turn_straight) > 0:
            # calculate f scores of all state machines
            pred_class_turn_straight, \
            true_class_turn_straight, \
            f1_turn_straight_macro, \
            f1_turn_straight_micro, \
            true_one_hot_turn_straight, \
            pred_one_hot_turn_straight, \
            brier_turn_straight = caclulate_scores(X_turn_straight, y_turn_straight, state_machine='turn_straight')

            # create qq plots
            qq_path = os.path.join(self.param_path, 'qq', str(step))
            if not os.path.exists(qq_path):
                os.makedirs(qq_path)
            true_turn = true_one_hot_turn_straight[:, 0]
            pred_turn = pred_one_hot_turn_straight[:, 0]
            create_qq_plot(y_true=true_turn, y_pred=pred_turn, title='turn')
            plt.savefig(os.path.join(qq_path, 'turn.pdf'))
            plt.close()
            true_straight = true_one_hot_turn_straight[:, 1]
            pred_straight = pred_one_hot_turn_straight[:, 1]
            create_qq_plot(y_true=true_straight, y_pred=pred_straight, title='straight')
            plt.savefig(os.path.join(qq_path, 'straight.pdf'))
            plt.close()

            # write summaries to tensorboard
            # write summaries
            summary_f1_turn_straight_micro = self.sess.run(self.summary_f1_turn_straight_micro,
                                                           feed_dict={self._scalar: f1_turn_straight_micro})
            self.file_writer.add_summary(summary_f1_turn_straight_micro, global_step=step)
            summary_f1_turn_straight_macro = self.sess.run(self.summary_f1_turn_straight_macro,
                                                           feed_dict={self._scalar: f1_turn_straight_macro})
            self.file_writer.add_summary(summary_f1_turn_straight_macro, global_step=step)
            summary_brier_turn_straight_macro = self.sess.run(self.summary_brier_score_turn_straight,
                                                              feed_dict={self._scalar: brier_turn_straight})
            self.file_writer.add_summary(summary_brier_turn_straight_macro, global_step=step)
            # plot confusion matrices
            conf_matrix_path = os.path.join(self.param_path, 'conufsion_matrices')
            if not os.path.exists(conf_matrix_path):
                os.makedirs(conf_matrix_path)
            conf_matrix_path_step = os.path.join(conf_matrix_path, str(step))
            if not os.path.exists(conf_matrix_path_step):
                os.makedirs(conf_matrix_path_step)
            plot_confusion_matrix(y_true=true_class_turn_straight, y_pred=pred_class_turn_straight,
                                  classes=['turn', 'straight'],
                                  title='Turn Straight', normalize=False)
            plt.savefig(os.path.join(conf_matrix_path_step, 'TurnStraight.pdf'))
            plt.close()
            plot_confusion_matrix(y_true=true_class_turn_straight, y_pred=pred_class_turn_straight,
                                  classes=['turn', 'straight'],
                                  title='Turn Straight', normalize=True)
            plt.savefig(os.path.join(conf_matrix_path_step, 'TurnStraightNorm.pdf'))
            plt.close()

        # left_right
        if len(X_left_right):
            # calculate f scores of all state machines
            pred_class_left_right, \
            true_class_left_right, \
            f1_left_right_macro, \
            f1_left_right_micro, \
            true_one_hot_left_right, \
            pred_one_hot_left_right, \
            brier_left_right = caclulate_scores(X_left_right, y_left_right, state_machine='left_right')

            # create qq plots
            qq_path = os.path.join(self.param_path, 'qq', str(step))
            if not os.path.exists(qq_path):
                os.makedirs(qq_path)
            true_left = true_one_hot_left_right[:, 0]
            pred_left = pred_one_hot_left_right[:, 0]
            create_qq_plot(y_true=true_left, y_pred=pred_left, title='left')
            plt.savefig(os.path.join(qq_path, 'left.pdf'))
            plt.close()
            true_right = true_one_hot_left_right[:, 1]
            pred_right = pred_one_hot_left_right[:, 1]
            create_qq_plot(y_true=true_right, y_pred=pred_right, title='right')
            plt.savefig(os.path.join(qq_path, 'right.pdf'))
            plt.close()

            # write summaries to tensorboard
            # write summaries
            summary_f1_left_right_micro = self.sess.run(self.summary_f1_left_right_micro,
                                                        feed_dict={self._scalar: f1_left_right_micro})
            self.file_writer.add_summary(summary_f1_left_right_micro, global_step=step)
            summary_f1_left_right_macro = self.sess.run(self.summary_f1_left_right_macro,
                                                        feed_dict={self._scalar: f1_left_right_macro})
            self.file_writer.add_summary(summary_f1_left_right_macro, global_step=step)
            summary_brier_left_right_macro = self.sess.run(self.summary_brier_score_left_right,
                                                           feed_dict={self._scalar: brier_left_right})
            self.file_writer.add_summary(summary_brier_left_right_macro, global_step=step)
            # plot confusion matrices
            conf_matrix_path = os.path.join(self.param_path, 'conufsion_matrices')
            if not os.path.exists(conf_matrix_path):
                os.makedirs(conf_matrix_path)
            conf_matrix_path_step = os.path.join(conf_matrix_path, str(step))
            if not os.path.exists(conf_matrix_path_step):
                os.makedirs(conf_matrix_path_step)
            plot_confusion_matrix(y_true=true_class_left_right, y_pred=pred_class_left_right,
                                  classes=['left', 'right'],
                                  title='Left Right', normalize=False)
            plt.savefig(os.path.join(conf_matrix_path_step, 'LeftRight.pdf'))
            plt.close()
            plot_confusion_matrix(y_true=true_class_left_right, y_pred=pred_class_left_right,
                                  classes=['left', 'right'],
                                  title='Left Right', normalize=True)
            plt.savefig(os.path.join(conf_matrix_path_step, 'LeftRightNorm.pdf'))
            plt.close()

        # start_stop_move
        if len(X_start_stop_move):
            # calculate f scores of all state machines
            pred_class_start_stop_move, \
            true_class_start_stop_move, \
            f1_start_stop_move_macro, \
            f1_start_stop_move_micro, \
            true_one_hot_start_stop_move, \
            pred_one_hot_start_stop_move, \
            brier_start_stop_move = caclulate_scores(X_start_stop_move, y_start_stop_move,
                                                     state_machine='start_stop_move')

            # create qq plots
            qq_path = os.path.join(self.param_path, 'qq', str(step))
            if not os.path.exists(qq_path):
                os.makedirs(qq_path)
            true_start = true_one_hot_start_stop_move[:, 0]
            pred_start = pred_one_hot_start_stop_move[:, 0]
            create_qq_plot(y_true=true_start, y_pred=pred_start, title='start')
            plt.savefig(os.path.join(qq_path, 'start.pdf'))
            plt.close()
            true_stop = true_one_hot_start_stop_move[:, 1]
            pred_stop = pred_one_hot_start_stop_move[:, 1]
            create_qq_plot(y_true=true_stop, y_pred=pred_stop, title='stop')
            plt.savefig(os.path.join(qq_path, 'stop.pdf'))
            plt.close()
            true_move = true_one_hot_start_stop_move[:, 2]
            pred_move = pred_one_hot_start_stop_move[:, 2]
            create_qq_plot(y_true=true_move, y_pred=pred_move, title='move')
            plt.savefig(os.path.join(qq_path, 'move.pdf'))
            plt.close()

            # write summaries to tensorboard
            # write summaries
            summary_f1_start_stop_move_micro = self.sess.run(self.summary_f1_start_stop_move_micro,
                                                             feed_dict={self._scalar: f1_start_stop_move_micro})
            self.file_writer.add_summary(summary_f1_start_stop_move_micro, global_step=step)
            summary_f1_start_stop_move_macro = self.sess.run(self.summary_f1_start_stop_move_macro,
                                                             feed_dict={self._scalar: f1_start_stop_move_macro})
            self.file_writer.add_summary(summary_f1_start_stop_move_macro, global_step=step)
            summary_brier_start_stop_move_macro = self.sess.run(self.summary_brier_score_start_stop_move,
                                                                feed_dict={self._scalar: brier_start_stop_move})
            self.file_writer.add_summary(summary_brier_start_stop_move_macro, global_step=step)
            # plot confusion matrices
            conf_matrix_path = os.path.join(self.param_path, 'conufsion_matrices')
            if not os.path.exists(conf_matrix_path):
                os.makedirs(conf_matrix_path)
            conf_matrix_path_step = os.path.join(conf_matrix_path, str(step))
            if not os.path.exists(conf_matrix_path_step):
                os.makedirs(conf_matrix_path_step)
            plot_confusion_matrix(y_true=true_class_start_stop_move, y_pred=pred_class_start_stop_move,
                                  classes=['start', 'stop', 'move'],
                                  title='Start Stop Move', normalize=False)
            plt.savefig(os.path.join(conf_matrix_path_step, 'StartStopMove.pdf'))
            plt.close()
            plot_confusion_matrix(y_true=true_class_start_stop_move, y_pred=pred_class_start_stop_move,
                                  classes=['start', 'stop', 'move'],
                                  title='Start Stop Move', normalize=True)
            plt.savefig(os.path.join(conf_matrix_path_step, 'StartStopMoveNorm.pdf'))
            plt.close()
        if self.mutex is not None:
            self.mutex.release()

    def predict(self, X=None, **kwargs):
        return self.predict_n_batches(X)

    def predict_n_batches(self, X, n_batches=None):
        if n_batches is not None:
            num_batches = n_batches
        elif self.batch_size != None:
            num_batches = int(len(X) // self.batch_size)
        else:
            num_batches = 0

        if num_batches == 0:
            images_hk1 = []
            images_hk2 = []
            trajectories = []
            for dic in X:
                if self.use_hk1:
                    images_hk1.append(dic['img_hk1'])
                if self.use_hk2:
                    images_hk2.append(dic['img_hk2'])
                if self.use_trajectory:
                    trajectories.append(dic['trajectory'])

            feed_dict = {self.training_flag: False}
            if self.use_hk1:
                feed_dict.update({self.input_hk1: np.array(images_hk1, dtype=np.float32) / 255.0})
            if self.use_hk2:
                feed_dict.update({self.input_hk2: np.array(images_hk2, dtype=np.float32) / 255.0})
            if self.use_trajectory:
                feed_dict.update({self.input_trajectory: trajectories})

            if self.train_wait_motion:
                pred_wait_motion = self.sess.run(self.prediction_wait_motion_proba, feed_dict=feed_dict)
            else:
                pred_wait_motion = np.array([0])
            if self.train_turn_straight:
                pred_turn_straight = self.sess.run(self.prediction_turn_straight_proba, feed_dict=feed_dict)
            else:
                pred_turn_straight = np.array([0])
            if self.train_left_right:
                pred_left_right = self.sess.run(self.prediction_left_right_proba, feed_dict=feed_dict)
            else:
                pred_left_right = np.array([0])
            if self.train_start_stop_move:
                pred_start_stop_move = self.sess.run(self.prediction_start_stop_move_proba, feed_dict=feed_dict)
            else:
                pred_start_stop_move = np.array([0])

            out = [pred_wait_motion.tolist(),
                   pred_turn_straight.tolist(),
                   pred_left_right.tolist(),
                   pred_start_stop_move.tolist()]
        else:
            out_wait_motion = []
            out_turn_straight = []
            out_left_right = []
            out_start_stop_move = []
            for batch in range(0, num_batches):
                batch_x = X[batch * self.batch_size:batch * self.batch_size + self.batch_size]

                ms_hk1 = []
                ms_hk2 = []
                of_hk1 = []
                of_hk2 = []
                trajectories = []
                for dic in batch_x:
                    if self.use_ms_hk1:
                        ms_hk1.append(dic['ms_hk1'])
                    if self.use_ms_hk2:
                        ms_hk2.append(dic['ms_hk2'])
                    if self.use_of_hk1:
                        of_hk1.append(dic['of_hk1'])
                    if self.use_of_hk2:
                        of_hk2.append(dic['of_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                feed_dict = {self.training_flag: False}
                if self.use_ms_hk1:
                    feed_dict.update({self.input_ms_hk1: np.swapaxes(
                        np.swapaxes(np.array(ms_hk1, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                if self.use_ms_hk2:
                    feed_dict.update({self.input_ms_hk2: np.swapaxes(
                        np.swapaxes(np.array(ms_hk2, dtype=np.float32), 1, 2), 2, 3) / 255.0})
                if self.use_of_hk1:
                    feed_dict.update({self.input_of_hk1: of_hk1})
                if self.use_of_hk2:
                    feed_dict.update({self.input_of_hk2: of_hk2})
                if self.use_trajectory:
                    feed_dict.update({self.input_trajectory: trajectories})

                if self.train_wait_motion:
                    pred_wait_motion = self.sess.run(self.prediction_wait_motion_proba, feed_dict=feed_dict)
                else:
                    pred_wait_motion = np.array([0])
                if self.train_turn_straight:
                    pred_turn_straight = self.sess.run(self.prediction_turn_straight_proba, feed_dict=feed_dict)
                else:
                    pred_turn_straight = np.array([0])
                if self.train_left_right:
                    pred_left_right = self.sess.run(self.prediction_left_right_proba, feed_dict=feed_dict)
                else:
                    pred_left_right = np.array([0])
                if self.train_start_stop_move:
                    pred_start_stop_move = self.sess.run(self.prediction_start_stop_move_proba, feed_dict=feed_dict)
                else:
                    pred_start_stop_move = np.array([0])

                out_wait_motion += pred_wait_motion.tolist()
                out_turn_straight += pred_turn_straight.tolist()
                out_left_right += pred_left_right.tolist()
                out_start_stop_move += pred_start_stop_move.tolist()
            # compute rest
            batch_x = X[batch * self.batch_size + self.batch_size:]
            if len(batch_x) > 0:
                images_hk1 = []
                images_hk2 = []
                trajectories = []
                for dic in batch_x:
                    if self.use_hk1:
                        images_hk1.append(dic['img_hk1'])
                    if self.use_hk2:
                        images_hk2.append(dic['img_hk2'])
                    if self.use_trajectory:
                        trajectories.append(dic['trajectory'])

                feed_dict = {self.training_flag: False}
                if self.use_hk1:
                    feed_dict.update({self.input_hk1: np.array(images_hk1, dtype=np.float32) / 255.0})
                if self.use_hk2:
                    feed_dict.update({self.input_hk2: np.array(images_hk2, dtype=np.float32) / 255.0})
                if self.use_trajectory:
                    feed_dict.update({self.input_trajectory: trajectories})

                if self.train_wait_motion:
                    pred_wait_motion = self.sess.run(self.prediction_wait_motion_proba, feed_dict=feed_dict)
                else:
                    pred_wait_motion = np.array([0])
                if self.train_turn_straight:
                    pred_turn_straight = self.sess.run(self.prediction_turn_straight_proba, feed_dict=feed_dict)
                else:
                    pred_turn_straight = np.array([0])
                if self.train_left_right:
                    pred_left_right = self.sess.run(self.prediction_left_right_proba, feed_dict=feed_dict)
                else:
                    pred_left_right = np.array([0])
                if self.train_start_stop_move:
                    pred_start_stop_move = self.sess.run(self.prediction_start_stop_move_proba, feed_dict=feed_dict)
                else:
                    pred_start_stop_move = np.array([0])
                out_wait_motion += pred_wait_motion.tolist()
                out_turn_straight += pred_turn_straight.tolist()
                out_left_right += pred_left_right.tolist()
                out_start_stop_move += pred_start_stop_move.tolist()

            out = [out_wait_motion,
                   out_turn_straight,
                   out_left_right,
                   out_start_stop_move]

        return out

    ## Creates tf graph of convolutional net
    # @param x (tensor): network input
    # @return output (network output)
    def _conv_net(self, x):
        if self.cnn_architecture == 'InceptionI3D':
            network = InceptionI3D(x=x, training=self.training_flag, use_batch_norm=False)
        return network.model


def model_loader(path, epoch='last'):
    import inspect
    params = pickle.load(open(path + '/hyperparameter.pkl', "rb"))
    if not epoch or epoch == 'last':
        epoch = int(pickle.load(open(path + '/last_epoch.pkl', "rb")))
    elif epoch == 'backup':
        epoch = None

    args, _, _, _ = inspect.getargspec(MSOFEstimatorStateMachine.__init__)
    params.update({'path': path[:path.rfind("/")]})
    params = {key: params[key] for key in args if key != 'self'}
    model = MSOFEstimatorStateMachine(**params)
    model.load(path, epoch=epoch)
    return model
