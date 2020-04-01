import argparse
import os
import numpy as np
import tqdm
import json
import pickle


## loads dataset split npy files from one folder and creates input memmap
# Warning! Large RAM needed for large dataset. The methods loads the complete set to memory before moving it to a
# memmap. This problem will be addressed in the soon.
#
# @param args (ArgumentParser()): see args in __main__
def main(args):
    used_validation_fold = args.used_validation_fold
    dataset_path = os.path.join(args.dataset_path, "dataset")
    datset_split_path = os.path.join(args.dataset_path, "dataset/split.json")
    files = os.listdir(dataset_path)
    dataset = {}
    print("loading dataset")
    # TODO: Add way to load dataset file by file. To do this, the complete dataset has to be loaded once to determine
    # TODO: the maximum length of a string. Afterwards the memmap can be created for the complete set and filled
    # TODO: file by file.
    # load all npy files into memory
    for f in tqdm.tqdm(files):
        if f.endswith('.npy'):
            data = np.load(os.path.join(dataset_path, f), allow_pickle=True, encoding='latin1').item()
            dataset.update(data)

    # preprocess and dataset and split into training and validation set
    from ImageSequenceDataset.PreprocessTrajectoryDataset.preprocess_set import PreprocessTrajectoryDataset
    pre = PreprocessTrajectoryDataset(split=datset_split_path, n_in=50,
                                      values_to_be_stored_input=['x_tracked', 'y_tracked', 'z_tracked',
                                                                 'ms_hk1',
                                                                 'ms_hk2',
                                                                 'of_hk1',
                                                                 'of_hk2'],
                                      values_to_be_stored_output=['tl', 'straight', 'tr', 'wait', 'start',
                                                                  'stop',
                                                                  'move'])

    D_train, D_test, Validation_Fold = pre.split_train_test(dataset)

    print('crop trajectories')
    X, y_train_one_array, validation_fold = pre.transform_set(D_train, use_ms=True, use_of=True)

    # create auxiliary classes
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

    # save labels
    y = np.array(y)
    np.save(os.path.join(dataset_path, "dataset_y.npy"), y)

    # create memmaps für train and validation
    train = []
    validation = []
    for i in range(len(X)):
        if validation_fold[i] == used_validation_fold:
            validation.append(i)
        else:
            train.append(i)

    json.dump({"train": train, "validation": validation},
              open(os.path.join(dataset_path, "train_validation.json"), 'w'))

    memmap_path_X_train = os.path.join(dataset_path, "memmap_X_train.dat")
    memmap_path_X_validation = os.path.join(dataset_path, "memmap_X_validation.dat")
    X = np.array(X)
    X_train = X[train]
    X_valid = X[validation]
    del X
    # train
    max_len = 0
    for i, x in enumerate(tqdm.tqdm(X_train)):
        X_train[i] = pickle.dumps(X_train[i])
        if len(X_train[i]) > max_len:
            # get max size of string to create memmap
            max_len = len(X_train[i])
    print(max_len)
    # save memmap configuration
    json.dump({"max_len": max_len}, open(os.path.join(dataset_path, "memmap_X_train.json"), 'w'))
    # add X_train to memmap
    fp_X = np.memmap(filename=memmap_path_X_train, mode="w+", dtype='|S' + str(max_len), shape=(len(X_train)))
    fp_X[:] = X_train[:]
    fp_X.flush()
    # validation
    max_len = 0
    for i, x in enumerate(tqdm.tqdm(X_valid)):
        X_valid[i] = pickle.dumps(X_valid[i])
        if len(X_valid[i]) > max_len:
            # get max size of string to create memmap
            max_len = len(X_valid[i])
    print(max_len)
    # save memmap configuration
    json.dump({"max_len": max_len}, open(os.path.join(dataset_path, "memmap_X_validation.json"), 'w'))
    # add X_valid to memmap
    fp_X = np.memmap(filename=memmap_path_X_validation, mode="w+", dtype='|S' + str(max_len), shape=(len(X_valid)))
    fp_X[:] = X_valid[:]
    fp_X.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', dest='dataset_path',
                        default='cyclist_action_recognition/',
                        help='Path to npy files and split.json.')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    args = parser.parse_args()
    main(args=args)
