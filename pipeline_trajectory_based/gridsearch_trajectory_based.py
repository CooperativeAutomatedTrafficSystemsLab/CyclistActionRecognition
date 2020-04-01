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
    used_validation_fold = 4
    network_path = args.network_path
    n_jobs = int(args.n_jobs)
    now = datetime.now()
    kwargs_json = {}
    # get training parameters from json file if provided
    if len(args.json_path) > 0:
        json_data = json.load(open(args.json_path, 'r'))
        param_grid = json_data['param_grid']
        iter_evaluate_list_dict(param_grid)
        kwargs_json = json_data['kwargs']
    # create output path to save training results to
    if len(args.continue_path) == 0:
        estimator_path = os.path.join(network_path, "trained_models") + now.strftime("/%Y_%m_%d__%H_%M_%S")
    else:
        estimator_path = args.continue_path
    # initialize  estimator
    from model.TrajectoryEstimatorStateMachine import TrajectoryEstimatorStateMachine
    kwargs = {"path": estimator_path, "save": False}
    kwargs.update(kwargs_json)
    estimator = TrajectoryEstimatorStateMachine(**kwargs)
    # load dataset and perform split and transformations
    dataset_path = os.path.join(network_path, "dataset/trajectory_dataset.npy")
    datset_split_path = os.path.join(network_path, "dataset/split.json")
    print("loading dataset")
    dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1').item()
    from ImageSequenceDataset.PreprocessTrajectoryDataset.preprocess_set import PreprocessTrajectoryDataset
    pre = PreprocessTrajectoryDataset(split=datset_split_path, n_in=50,
                                      values_to_be_stored_input=['x_tracked', 'y_tracked', 'z_tracked'],
                                      values_to_be_stored_output=['tl', 'straight', 'tr', 'wait', 'start',
                                                                  'stop',
                                                                  'move'])

    D_train, D_test, Validation_Fold = pre.split_train_test(dataset)

    print('crop trajectories')
    X, y_train_one_array, validation_fold = pre.transform_set(D_train)
    X = np.concatenate(X, axis=0)

    # create network outputs
    y = []
    for sample in y_train_one_array:
        labels_wait_motion = [sample[3], max(sample[4:])]
        labels_turn_straight = [max(sample[0], sample[2]), sample[1]]
        labels_left_right = [sample[0], sample[2]]
        labels_start_stop_move = sample[4:]
        if sum(labels_wait_motion) != 1:
            labels_wait_motion = [0, 0]
        if sum(labels_turn_straight) != 1:
            labels_turn_straight = [0, 0]
        if sum(labels_left_right) != 1:
            labels_left_right = [0, 0]
        if sum(labels_start_stop_move) != 1:
            labels_start_stop_move = [0, 0, 0]

        y.append(
            {"labels_wait_motion": labels_wait_motion,
             "labels_turn_straight": labels_turn_straight,
             "labels_left_right": labels_left_right,
             "labels_start_stop_move": labels_start_stop_move})

    # create train and validation indices
    train = []
    validation = []
    for i in range(len(validation_fold)):
        if validation_fold[i] == used_validation_fold:
            validation.append(i)
        else:
            train.append(i)

    # perform grid search
    gridSearch = TFGridSearch(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs, save_path=estimator_path,
                              verbose=10)

    gridSearch.fit(X=np.array(X),
                   y=np.array(y),
                   train=train,
                   validation=validation,
                   steps=100000,
                   print_steps=10,
                   score_steps=250,
                   continue_gridSearch=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument('--networkPath', dest='network_path',
                        default='/media/ssd/stefan/basic_movement_detection/trajectory/',
                        help='Path to network folder including dataset')
    parser.add_argument('--n_jobs', dest='n_jobs', default=5,
                        help='Number parallel trained models')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    parser.add_argument('--cuda_devices', dest='cuda_devices', default='0',
                        help='Which gpus to use.')
    parser.add_argument('--json_path', dest='json_path', default='./sweeps/turn_straight_0.json',
                        help='Path to json with network parameters')
    parser.add_argument('--continue_path', dest='continue_path',
                        default='',
                        help='If path is set, grid search will try to resume last training in the passed folder')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    main(args=args)
