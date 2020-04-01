from multiprocessing import Process, Manager
import numpy as np
import time
import json
import os
import pickle
import copy
import shutil
from .ParameterGrid import ParameterGrid, _check_param_grid


## A object to perform a parallelized grid search with cross validation for models inherited from basic_network
#
class TFGridSearch(object):
    ## Constructor
    #   @param estimator: the model inherited from basic_network
    #   @param param_grid: the grid to be swept
    #   @param scoring:     Use scoring
    #   @param save_path:   the path where the grid search will be saved (this save_path is dominant over the save_path
    #                       in the BasicNetwork)
    #   @param n_jobs:      number models trained parallel (starts a new process for every project)
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=1, save_path=''):

        self.param_grid = param_grid
        #   check parameter grid for validity
        _check_param_grid(param_grid)
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = 'n_jobs'
        if save_path == '':
            raise ValueError('No save path is set!')
        if 'final' in save_path.split(os.sep):
            raise ValueError('Save path should not contain a directory named "final"!')
        self.save_path = save_path
        self.use_shared_memory = estimator.use_shared_memory
        self.seed = estimator.seed
        self.mutex = estimator.mutex

    ## destructor
    #  used to delete ppid_gpu file after script is killed
    def __del__(self):
        filename = str(os.getppid()) + "_gpu_processes"
        lock_filename = str(os.getppid()) + "_gpu_processes.lock"
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(lock_filename):
            os.remove(lock_filename)

    ##  Gets iterator from parameter grid
    #   @return:    iterator
    def _get_param_iterator(self):
        return ParameterGrid(self.param_grid)

    ## Core fit function
    # Performs a GridSearch with cross validation for given estimator and given parameter grid
    # @param X:             Dataset Input
    # @param y:             Dataset Ground Truth
    # @param train:               indices of train-set X[train], y[train]
    # @param validation:          indices of valid-set X[validation], y[validation]
    # @param steps:         number epochs for training
    # @param print_steps:   prints the train loss every n'th epoch
    # @param score_steps:   calculates score on validation data every n'th epoch
    # @param cyclic_steps:  calls cyclic method on validation data every n'th epoch
    # @param continue_gridSearch:  boolean to decide wether a grid_search should be continued or not
    # @param **kwargs:      variable number of keyword arguments
    def fit(self, X, y=None, train=None, validation=None, steps=None, print_steps=None, score_steps=None,
            cyclic_steps=None,
            continue_gridSearch=False, **kwargs):

        if continue_gridSearch:
            path_to_canceled = self.save_path
        else:
            path_to_canceled = None

        # regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())

        n_candidates = len(candidate_params)
        print("[GridSearch] Fitting one model for each of " + str(n_candidates) + " candidates")

        if self.use_shared_memory:
            # create a file with name of parent pid as name, init [0]*NUMBER_OF_GPUS_USED
            # get parent pid (same in every subprocess or thread)
            ppid = os.getppid()
            print("ppid", ppid)
            # get used gpus or default to 0
            if os.environ["CUDA_VISIBLE_DEVICES"]:
                available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                available_devices = '0'

            available_devices = available_devices.split(",")
            # save to with distinct filename (unique by use of ppid)
            json.dump([0] * len(available_devices), open(str(ppid) + "_gpu_processes", "w"))

        # core loop for training the individual models
        # the loop is parallelized using the job-lib
        # for every model configuration a own python process is stared
        # if there is a path to a canceled grid search, continue the grid search at the last known point
        if path_to_canceled:
            parameters_to_sweep = self.get_param_config(candidate_params)

            n_folds = len(parameters_to_sweep)
            folds_running = [0] * n_folds
            processes = {}
            manager = Manager()
            out = manager.list()
            while True:
                # check if new process has to be started
                # count processes running
                count = folds_running.count(1)
                if count < self.n_jobs and 0 in folds_running:
                    # start new process and add it to folds_running
                    # get idx of process to start
                    idx = next(idx for idx, sweep in enumerate(folds_running) if sweep == 0)

                    parameters = parameters_to_sweep[idx]
                    args = {'path_to_canceled': path_to_canceled, 'estimator': copy.deepcopy(self.estimator), 'X': X,
                            'y': y, 'parameters': parameters, 'train': train,
                            'validation': validation, 'steps': steps,
                            'print_steps': print_steps, 'return_list': out,
                            'score_steps': score_steps, 'cyclic_steps': cyclic_steps}
                    args.update(kwargs)
                    # start the process
                    p = Process(target=_continue_fit, kwargs=args)
                    p.start()
                    processes.update({idx: p})
                    folds_running[idx] = 1
                # check if processes are finished
                for key in list(processes.keys()):
                    if not processes[key].is_alive():
                        processes[key].join()
                        processes.pop(key)
                        folds_running[key] = 2
                # check if processing is finished
                done = True
                for item in folds_running:
                    if item != 2:
                        done = False
                        break
                if done:
                    break
                time.sleep(50)
        # if there is no path set, start a new grid search
        else:
            parameters_to_sweep = self.get_param_config(candidate_params)

            n_folds = len(parameters_to_sweep)
            folds_running = [0] * n_folds
            processes = {}
            manager = Manager()
            out = manager.list()
            while True:
                # check if new process has to be started
                # count processes running
                count = folds_running.count(1)
                if count < self.n_jobs and 0 in folds_running:
                    # start new process and add it to folds_running
                    # get idx of process to start
                    idx = next(idx for idx, sweep in enumerate(folds_running) if sweep == 0)

                    parameters = parameters_to_sweep[idx]
                    args = {'estimator': copy.deepcopy(self.estimator), 'X': X, 'y': y, 'parameters': parameters,
                            'train': train,
                            'validation': validation, 'steps': steps,
                            'print_steps': print_steps, 'return_list': out,
                            'score_steps': score_steps, 'cyclic_steps': cyclic_steps}
                    args.update(kwargs)
                    # start the process
                    p = Process(target=_fit, kwargs=args)
                    p.start()
                    processes.update({idx: p})
                    folds_running[idx] = 1
                # check if processes are finished
                for key in list(processes.keys()):
                    if not processes[key].is_alive():
                        processes[key].join()
                        processes.pop(key)
                        folds_running[key] = 2
                # check if processing is finished
                done = True
                for item in folds_running:
                    if item != 2:
                        done = False
                        break
                if done:
                    break
                time.sleep(50)

        fold_keys = []
        for fold in out:
            fold_keys.append(fold[0])

        all_params = {}  # dict to store all parameter configuration
        all_epochs = {}  # dict to store all epochs
        all_scores = {}  # dict to store all scores

        # find all unique keys representing the individual configurations
        unique_keys = np.unique(np.array(fold_keys))
        # loop over all unique keys
        for key in unique_keys:
            for o in out:
                if o[0] == key:
                    # get configuration name
                    params = o[1]
                    # get scores
                    score_sum = o[2]
                    # get epochs, ensure int
                    epochs = list(map(int, o[3]))

            all_params.update({key: params})
            all_epochs.update({key: epochs})
            all_scores.update({key: score_sum})

        print('[GridSearch] Find best configuration and best epoch')

        # find best key and step
        max_score = -1e5  # low default value since the score should be as high as possible
        best_config = None  # default value for the best configuration
        best_step_idx = 0  # default idx for the best step

        # loop over all items in the avrg_scores dict containing the configuration an the scores averaged over the folds
        for config, scores in all_scores.items():
            score = None
            if steps in all_epochs[config]:
                # find the highest score for one configuration
                score = np.nanmax(scores)
                # get the step associated with the biggest score
                if not np.isnan(score):
                    step_idx = scores.index(score)

            else:
                print('WARNING: there are no scores for the config: ', config)

            # perform a local hill climber tho find the biggest score over all configurations
            if score and score > max_score:
                max_score = score
                best_config = config
                best_step_idx = step_idx

        if best_config is None:
            raise ValueError('No best configuration found! Cannot train an final estimator')

        # get the parameter of the best configuration
        best_params = all_params[best_config]
        # get the epoch of the biggest score of the best configuration
        self.best_epoch = int(all_epochs[best_config][best_step_idx])

        print('[GridSearch] Save GridSearch results in ' + os.path.join(self.save_path, "grid_search_output.json"))
        # write output to json file
        json.dump(
            {"all": [(i[2], i[3], i[0]) for i in out]},
            open(os.path.join(self.save_path, "grid_search_output.json"), "w"))

        time.sleep(60)

        print('[GridSearch] Train Final Estimator')

        # check if a final estimator exists
        if os.path.isdir(best_params["path"] + "/final/"):
            i = 0
            old_final_path = best_params["path"] + "/final_old_" + str(i) + "/"
            # handle the case if multiple final models were trained
            while os.path.isdir(best_params["path"] + "/final_old_" + str(i) + "/"):
                i += 1
                old_final_path = best_params["path"] + "/final_old_" + str(i) + "/"
            print('WARNING: a final model was already trained and will be moved to "' + old_final_path + '"')
            # move final model to new path
            shutil.copytree(best_params["path"] + "/final/", old_final_path)
            # remove dir of final model
            shutil.rmtree(best_params["path"] + "/final/")

        # clone base estimator
        final_estimator = copy.deepcopy(self.estimator)
        # set path for final
        best_params["path"] += "/final/"
        # set save to true
        best_params["save"] = True

        # set the params of the best configurations as the params of the final estimator
        final_estimator.set_params(**best_params)
        # train the final estimator on the complete train set, until the best epoch
        final_estimator.fit(X=X, y=y, steps=self.best_epoch, print_steps=print_steps, **kwargs)

        self.best_modelpath = final_estimator.path + final_estimator.hash_name
        print('[GridSearch] Finished. Best Model: ', self.best_modelpath, ' at epoch: ', self.best_epoch)

    ## Creates parameter config list
    # @param candidate_params:  candidate params for the grid search
    # @return:                  list containing params
    def get_param_config(self, candidate_params):
        tmp_params = []
        for par in candidate_params:
            param_dict = dict(par)
            param_dict.update({"path": self.save_path})
            param_dict.update({"seed": self.seed})
            param_dict.update({"use_shared_memory": self.use_shared_memory})
            param_dict.update({"mutex": self.mutex})
            tmp_params.append(param_dict)
        return tmp_params


def _fit(estimator, X, y, parameters, train=None, validation=None, steps=None, print_steps=None, score_steps=None,
         cyclic_steps=None, return_list=None, **kwargs):
    if parameters is not None:
        msg = '%s' % (', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

        # add path as key
        estimator.set_params(**parameters)
        estimator.fit(X=X, y=y, train=train, validation=validation, scoring=True, steps=steps,
                      print_steps=print_steps, score_steps=score_steps, cyclic_steps=cyclic_steps, **kwargs)

        # return the score dict of the estimator
        score_file = os.path.join(estimator.path, estimator.hash_name, 'backup', 'scores.pkl')
        with open(score_file, 'rb') as s_file:
            os.fsync(s_file)
            scores = pickle.load(s_file)
        score_dict = estimator.get_score_dict()
        scores = np.array([[epoch, scores[epoch]] for epoch in sorted(scores)])
        score_dict[2] = list(scores[:, 1])  # scores
        score_dict[3] = list(scores[:, 0])  # epochs
        return_list.append(score_dict)
    else:
        raise ValueError('Cannot train an estimator without parameters')


def _continue_fit(path_to_canceled, estimator, X, y, parameters, train=None, validation=None, steps=None,
                  print_steps=None, score_steps=None, cyclic_steps=None, return_list=None, **kwargs):
    if parameters is not None:
        # stitch parameter string
        msg = '%s' % (', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        is_valid_config = False
        # search for a previously trained model with the given parameter configuration
        # loop over all previously trained models
        for root, dirs, files in os.walk(path_to_canceled):
            same_value_lst = []
            # if there is a backup and its not the final estimator, load the hyperparameter
            if 'hyperparameter.pkl' in files and 'backup' in dirs and 'final' not in sum(
                    map(lambda x: str(x).split('_'), root.split(os.sep)), []):
                parameters_canceled = pickle.load(open(os.path.join(root, 'hyperparameter.pkl'), 'rb'),
                                                  encoding='latin1')
                # compare the values of the loaded hyperparameters with the parameters to be set
                for key in parameters.keys():
                    if key not in ['seed', 'mutex', 'use_shared_memory']:
                        if key == 'path' or parameters[key] == parameters_canceled[key]:
                            same_value_lst.append(True)
                        else:
                            same_value_lst.append(False)
                # if they are the same, a previously trained model with the given parameter configuration is found
                if all(same_value_lst) and len(same_value_lst) > 0:
                    is_valid_config = True
                    path = root
                    break

        if is_valid_config:
            # if a previously trained model with the given parameter configuration was found
            # load this model and continue to fit
            # lode epoch from backup
            epoch_file = os.path.join(path, 'backup', 'epoch.pkl')
            with open(epoch_file, 'rb') as s_file:
                os.fsync(s_file)
                start_epoch = int(pickle.load(s_file))

            if start_epoch + 1 >= steps:
                print("[CV] model already trained %s %s" % (msg, (64 - len(msg)) * '.'))
                # update score dict parameter
                score_dict = estimator.get_score_dict()
                for key in parameters.keys():
                    estimator.get_score_dict()[1]['key'] = parameters[key]
            else:
                print("[CV] continue %s %s" % (msg, (64 - len(msg)) * '.'))
                estimator.continue_fit(X=X, y=y, train=train, validation=validation, scoring=True, steps=steps,
                                       print_steps=print_steps, score_steps=score_steps, cyclic_steps=cyclic_steps,
                                       path_to_canceled=path, **kwargs)
                score_dict = estimator.get_score_dict()
            score_file = os.path.join(path, 'backup', 'scores.pkl')
        else:
            # else start with a new model
            print("[CV] start %s %s" % (msg, (64 - len(msg)) * '.'))
            estimator.set_params(**parameters)
            estimator.fit(X=X, y=y, train=train, validation=validation, scoring=True, steps=steps,
                          print_steps=print_steps, score_steps=score_steps, cyclic_steps=cyclic_steps,
                          **kwargs)
            score_dict = estimator.get_score_dict()
            score_file = os.path.join(estimator.path, estimator.hash_name, 'backup', 'scores.pkl')

        with open(score_file, 'rb') as s_file:
            os.fsync(s_file)
            scores = pickle.load(s_file)
        scores = np.array([[epoch, scores[epoch]] for epoch in sorted(scores)])
        score_dict[2] = list(scores[:, 1])  # scores
        score_dict[3] = list(scores[:, 0])  # epochs
        return_list.append(score_dict)
    else:
        raise ValueError('Cannot train an estimator without parameters')
