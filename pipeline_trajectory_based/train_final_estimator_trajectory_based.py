import argparse
import os
import numpy as np
from datetime import datetime
import json


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
    used_validation_fold = args.used_validation_fold
    network_path = args.network_path
    now = datetime.now()
    kwargs_json = {}
    # get training parameters from json file if provided
    if len(args.json_path) > 0:
        json_data = json.load(open(args.json_path, 'r'))
        kwargs_json = json_data['kwargs']
        iter_evaluate_list_dict(kwargs_json)
    # create output path to save training results to
    if len(args.continue_path) == 0:
        estimator_path = os.path.join(network_path, "trained_models/") + json_data['name'] + "_" + now.strftime(
            "/%Y_%m_%d__%H_%M_%S")
    else:
        estimator_path = args.continue_path
    # initialize  estimator
    from model.TrajectoryEstimatorStateMachine import TrajectoryEstimatorStateMachine
    kwargs = {"path": estimator_path, "save": True}
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

    # fit estimator
    estimator.fit(X=np.array(X),
                  y=np.array(y),
                  train=train,
                  validation=validation,
                  print_steps=250,
                  scoring=True,
                  steps=int(args.steps),
                  score_steps=int(args.score_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument('--networkPath', dest='network_path',
                        default='trajectory_based/',
                        help='Path to network folder including dataset')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    parser.add_argument('--cuda_devices', dest='cuda_devices', default='0',
                        help='Which gpus to use.')
    parser.add_argument('--json_path', dest='json_path', default='./final/final_wait_motion.json',
                        help='Path to json with network parameters')
    parser.add_argument('--continue_path', dest='continue_path',
                        default='',
                        help='If path is set, grid search will try to resume last training in the passed folder')
    parser.add_argument('--steps', dest='steps', default=50000,
                        help='Number of train steps')
    parser.add_argument('--score_steps', dest='score_steps', default=250,
                        help='Number of score_steps')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    main(args=args)
