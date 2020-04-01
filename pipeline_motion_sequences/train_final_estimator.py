import argparse
import os
import numpy as np
from datetime import datetime
import json
import tf


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
    now = datetime.now()
    json_data = {}
    name = ""
    # get training parameters from json file if provided
    if len(args.json_path) > 0:
        json_data = json.load(open(args.json_path, 'r'))
        param_grid = json_data['param_grid']
        name = json_data['model_name']
    # create output path to save training results to
    if len(args.continue_path) == 0:
        estimator_path = os.path.join(network_path, "trained_models") + now.strftime("/%Y_%m_%d__%H_%M_%S_") + name
    else:
        estimator_path = args.continue_path
    # initialize  estimator
    from model.MSOFEstimatorStateMachineTransferMemmap import MSOFEstimatorStateMachine
    kwargs = {"path": estimator_path[:estimator_path.rfind("/")],
              "save": True,
              "batch_size": 27,
              "use_shared_memory": False,
              "train_turn_straight": False,
              "train_wait_motion": True}
    if "kwargs" in json_data.keys():
        kwargs.update(json_data["kwargs"])
        kwargs.update(param_grid)
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

    X = {'X_train': X_train,
         'X_validation': X_validation}

    # fit estimator
    if len(args.continue_path) == 0:
        estimator.fit(X=X,
                      y=y,
                      train=train,
                      validation=validation,
                      print_steps=5,
                      scoring=True,
                      steps=int(args.steps),
                      score_steps=int(args.score_steps))
    else:
        estimator.continue_fit(X=X, y=y, train=train, validation=validation, scoring=True, steps=int(args.steps),
                               print_steps=5, score_steps=int(args.score_steps),
                               path_to_canceled=args.continue_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument('--networkPath', dest='network_path',
                        default='motion_sequence_based/',
                        help='Path to training images')
    parser.add_argument('--n_jobs', dest='n_jobs', default=1,
                        help='Number parallel trained models')
    parser.add_argument('--steps', dest='steps', default=10000,
                        help='Number of train steps')
    parser.add_argument('--score_steps', dest='score_steps', default=200,
                        help='Number of score_steps')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    parser.add_argument('--cuda_devices', dest='cuda_devices', default='0',
                        help='Which gpus to use.')
    parser.add_argument('--json_path', dest='json_path', default='./final/wait_motion_final.json',
                        help='Path to json with network parameters')
    parser.add_argument('--continue_path', dest='continue_path',
                        default='',
                        help='If path is set, grid search will try to resume last training in the passed folder')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    main(args=args)
