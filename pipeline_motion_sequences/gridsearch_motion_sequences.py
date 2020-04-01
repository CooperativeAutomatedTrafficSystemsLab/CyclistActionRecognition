import argparse
import os
import numpy as np
from datetime import datetime
import json
from LF_Python3.tf_GridSearch_multiprocessing import TFGridSearch


## Iterates over nested dict and performs eval for every string in dict that contains the word "eval"
# ("eval" is removed from string before eval is performed)
# e.g.: dic[key]: "evaltf.nn.elu" -> dic[key] = eval("tf.nn.elu)
#
# @param nested (dict): dictionary containing param dict with function to call using eval
def iter_evaluate_list_dict(nested):
    if type(nested) == list:
        for item in nested:
            if type(item) == list or type(item) == dict:
                iter_evaluate_list_dict(nested=item)
    elif type(nested) == dict:
        for key in nested.keys():
            item = nested[key]
            if type(item) == list or type(item) == dict:
                iter_evaluate_list_dict(nested=item)
            elif type(item) == str or type(item) == bytes:
                if "eval" in item:
                    code = item[4:]
                    nested[key] = eval(code)


def main(args):
    network_path = args.network_path
    n_jobs = int(args.n_jobs)
    now = datetime.now()
    json_data = {}
    # get sweep parameters from json file if provided
    if len(args.json_path) > 0:
        json_data = json.load(open(args.json_path, 'r'))
        param_grid = json_data['param_grid']
    # create output path to save training results to
    if len(args.continue_path) == 0:
        estimator_path = os.path.join(network_path, "trained_models") + now.strftime("/%Y_%m_%d__%H_%M_%S")
    else:
        estimator_path = args.continue_path
    # initialize  estimator
    from model.MSOFEstimatorStateMachineTransferMemmap import MSOFEstimatorStateMachine
    kwargs = {"path": estimator_path,
              "save": False,
              "batch_size": 27,
              "use_shared_memory": False,
              "use_transfer_learning": True,
              "train_turn_straight": False,
              "train_wait_motion": True}
    if "kwargs" in json_data.keys():
        kwargs.update(json_data["kwargs"])
    estimator = MSOFEstimatorStateMachine(**kwargs)
    # load dataset as memmap
    dataset_path = os.path.join(network_path, "dataset")
    max_len = json.load(open(os.path.join(dataset_path, "memmap_X_train.json"), 'r'))['max_len']
    memmap_path_X = os.path.join(dataset_path, "memmap_X_train.dat")
    X_train = np.memmap(filename=memmap_path_X, mode="r", dtype='|S' + str(max_len))
    max_len = json.load(open(os.path.join(dataset_path, "memmap_X_validation.json"), 'r'))['max_len']
    memmap_path_X = os.path.join(dataset_path, "memmap_X_validation.dat")
    X_validation = np.memmap(filename=memmap_path_X, mode="r", dtype='|S' + str(max_len))
    y = np.load(os.path.join(dataset_path, "dataset_y.npy"), allow_pickle=True)
    train_validation = json.load(open(os.path.join(dataset_path, "train_validation.json"), 'r'))
    train = train_validation['train']
    validation = train_validation['validation']

    # perform gridsearch
    gridSearch = TFGridSearch(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs, save_path=estimator_path,
                              verbose=10)

    X = {'X_train': X_train,
         'X_validation': X_validation}
    gridSearch.fit(X=X,
                   y=y,
                   train=train,
                   validation=validation,
                   steps=50000,
                   print_steps=10,
                   score_steps=250,
                   continue_gridSearch=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument('--networkPath', dest='network_path',
                        default='motion_sequence_based/',
                        help='Path to training images')
    parser.add_argument('--n_jobs', dest='n_jobs', default=1,
                        help='Number parallel trained models')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    parser.add_argument('--cuda_devices', dest='cuda_devices', default='1',
                        help='Which gpus to use.')
    parser.add_argument('--json_path', dest='json_path', default='./sweeps/wait_motion.json',
                        help='Path to json with network parameters')
    parser.add_argument('--continue_path', dest='continue_path',
                        default='',
                        help='If path is set, grid search will try to resume last training in the passed folder')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    main(args=args)
