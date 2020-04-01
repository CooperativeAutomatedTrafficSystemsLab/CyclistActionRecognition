import argparse
import os
import numpy as np
import tqdm
import time
import pickle
import cv2 as cv
import zlib
import gc
from ImageSequenceDataset.PreprocessTrajectoryDataset.preprocess_set import PreprocessTrajectoryDataset


## Creates probabilities for all test scenes using trained estimator
# network_path should point to the path where trained_models/final/NAME_STATE_MACHINE is located
# (model.ckpt files needed9
# outputs are written to trained_models/final/NAME_STATE_MACHINE/probs.npy
#
def main(args):
    network_path = args.network_path
    estimator_paths = {"wait_motion": os.path.join(network_path, "trained_models/final/wait_motion"),
                       "turn_straight": os.path.join(network_path, "trained_models/final/turn_straight"),
                       "left_right": os.path.join(network_path, "trained_models/final/left_right"),
                       "start_stop_move": os.path.join(network_path, "trained_models/final/start_stop_move")}
    # set path to state machine
    estimator_name = args.state_machine
    output_path = os.path.join(estimator_paths[estimator_name], "probs.npy")
    # set parameters parameters and load trained model
    from model.MSOFEstimatorStateMachineTransferMemmap import MSOFEstimatorStateMachine
    hyper_param_path = os.path.join(estimator_paths[estimator_name], 'hyperparameter.pkl')
    hyper_params = pickle.load(open(hyper_param_path, 'rb'), encoding='latin1')
    kwargs = {"train_wait_motion": False,
              "train_turn_straight": False,
              "train_left_right": False,
              "train_start_stop_move": False}
    kwargs["train_" + estimator_name] = True
    kwargs.update(hyper_params)
    kwargs.pop('n_inputs')
    kwargs.pop('n_outputs')
    kwargs.pop('grad_clip')
    kwargs.pop('basic_params')
    estimator = MSOFEstimatorStateMachine(**kwargs)
    save_last = os.path.join(estimator_paths[estimator_name], 'model.ckpt')
    estimator.build_graph()
    estimator.saver.restore(estimator.sess, save_last)
    dataset_path = os.path.join(network_path, "dataset")
    datset_split_path = os.path.join(network_path, "dataset/split.json")
    files = os.listdir(dataset_path)

    print("loading dataset")
    # if output file already exists, start where the processing left off, otherwise create new output file
    if os.path.exists(output_path):
        output_dict = np.load(output_path, allow_pickle=True, encoding='latin1').item()
    else:
        output_dict = {}
    # go through all numpy files and process test scenes
    for f in tqdm.tqdm(files, desc="files"):
        if f.endswith('.npy'):
            # load dataset and perform transformations
            data = np.load(os.path.join(dataset_path, f), allow_pickle=True, encoding='latin1').item()

            pre = PreprocessTrajectoryDataset(split=datset_split_path, n_in=50,
                                              values_to_be_stored_input=['x_tracked', 'y_tracked', 'z_tracked',
                                                                         'ms_hk1',
                                                                         'ms_hk2',
                                                                         'of_hk1',
                                                                         'of_hk2'],
                                              values_to_be_stored_output=['tl', 'straight', 'tr', 'wait', 'start',
                                                                          'stop',
                                                                          'move'])

            _, D_test, Validation_Fold = pre.split_train_test(data)

            print('crop trajectories')

            # go through all scenes and predict probabilities for every sample
            for name in tqdm.tqdm(list(D_test.keys()), desc="scenes"):
                if name not in list(output_dict.keys()):
                    # load samples from scene and perform transformations
                    scene_dict = {}
                    temp_dict = {name: D_test[name]}
                    scene_dict.update({"x_tracked": temp_dict[name]["x_tracked"],
                                       "y_tracked": temp_dict[name]["y_tracked"],
                                       "z_tracked": temp_dict[name]["z_tracked"],
                                       "x_smoothed": temp_dict[name]["x_smoothed"],
                                       "y_smoothed": temp_dict[name]["y_smoothed"],
                                       "z_smoothed": temp_dict[name]["z_smoothed"],
                                       "orientation": temp_dict[name]["orientation"],
                                       "wait": temp_dict[name]["wait"],
                                       "start": temp_dict[name]["start"],
                                       "stop": temp_dict[name]["stop"],
                                       "move": temp_dict[name]["move"],
                                       "straight": temp_dict[name]["straight"],
                                       "tl": temp_dict[name]["tl"],
                                       "tr": temp_dict[name]["tr"],
                                       "ts": temp_dict[name]["ts"],
                                       "labels_wait_motion": [],
                                       "pred_wait_motion": [],
                                       "labels_turn_straight": [],
                                       "pred_turn_straight": [],
                                       "labels_left_right": [],
                                       "pred_left_right": [],
                                       "labels_start_stop_move": [],
                                       "pred_start_stop_move": [],
                                       })
                    X, y_train_one_array, validation_fold = pre.transform_set(temp_dict,
                                                                              use_ms=False,
                                                                              use_of=True)
                    # create ground truth (not needed for processing, but is saved for later evaluation)
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

                        scene_dict["labels_wait_motion"].append(np.array(labels_wait_motion))
                        scene_dict["labels_turn_straight"].append(np.array(labels_turn_straight))
                        scene_dict["labels_left_right"].append(np.array(labels_left_right))
                        scene_dict["labels_start_stop_move"].append(np.array(labels_start_stop_move))
                        y.append(
                            {"labels_wait_motion": labels_wait_motion,
                             "labels_turn_straight": labels_turn_straight,
                             "labels_left_right": labels_left_right,
                             "labels_start_stop_move": labels_start_stop_move})

                    # perform inference for every sample in scene
                    mean_time = 0
                    time_cnt = 0
                    for i in range(len(X)):
                        feed_dict = {estimator.training_flag: False}
                        if estimator.use_ms_hk1:
                            ms_hk1 = [cv.cvtColor(cv.imdecode(image, 1), cv.COLOR_BGR2GRAY) for image in
                                      X[i]['ms_hk1']]
                            feed_dict.update({estimator.input_ms_hk1: np.swapaxes(
                                np.swapaxes(np.array([ms_hk1], dtype=np.float32), 1, 2), 2, 3) / 255.0})
                        if estimator.use_ms_hk2:
                            ms_hk2 = [cv.cvtColor(cv.imdecode(image, 1), cv.COLOR_BGR2GRAY) for image in
                                      X[i]['ms_hk2']]
                            feed_dict.update({estimator.input_ms_hk2: np.swapaxes(
                                np.swapaxes(np.array([ms_hk2], dtype=np.float32), 1, 2), 2, 3) / 255.0})
                        if estimator.use_of_hk1:
                            of_hk1 = pickle.loads(zlib.decompress(X[i]['of_hk1'])).astype(
                                np.float16) * 2.0 / 255.0 - 1.0
                            of_hk1 = np.concatenate(of_hk1, axis=-1)
                            feed_dict.update({estimator.input_of_hk1: [of_hk1]})
                        if estimator.use_of_hk2:
                            of_hk2 = pickle.loads(zlib.decompress(X[i]['of_hk2'])).astype(
                                np.float16) * 2.0 / 255.0 - 1.0
                            of_hk2 = np.concatenate(of_hk2, axis=-1)
                            feed_dict.update({estimator.input_of_hk2: [of_hk2]})
                        if estimator.use_trajectory:
                            trajectory = X[i]['trajectory']
                            feed_dict.update({estimator.input_trajectory: [trajectory]})
                        start_time = time.time()
                        if estimator_name == "wait_motion":
                            prediction = estimator.sess.run(estimator.prediction_wait_motion_proba, feed_dict=feed_dict)
                        elif estimator_name == "turn_straight":
                            prediction = estimator.sess.run(estimator.prediction_turn_straight_proba,
                                                            feed_dict=feed_dict)
                        elif estimator_name == "left_right":
                            prediction = estimator.sess.run(estimator.prediction_left_right_proba, feed_dict=feed_dict)
                        elif estimator_name == "start_stop_move":
                            prediction = estimator.sess.run(estimator.prediction_start_stop_move_proba,
                                                            feed_dict=feed_dict)

                        if i != 0:
                            mean_time += time.time() - start_time
                            time_cnt += 1

                        if estimator_name == "wait_motion":
                            scene_dict["pred_wait_motion"].append(prediction[0])
                        elif estimator_name == "turn_straight":
                            scene_dict["pred_turn_straight"].append(prediction[0])
                        elif estimator_name == "left_right":
                            scene_dict["pred_left_right"].append(prediction[0])
                        elif estimator_name == "start_stop_move":
                            scene_dict["pred_start_stop_move"].append(prediction[0])

                    # show mean inference time per scene and add scene to output file
                    mean_time /= float(time_cnt)
                    print(mean_time)
                    output_dict.update({name: scene_dict})
                    np.save(output_path, output_dict)
            # free memory
            del data
            del D_test
            del pre
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument('--networkPath', dest='network_path',
                        default='motion_sequences_based/',
                        help='Path to training images')
    parser.add_argument('--used_validation_fold', dest='used_validation_fold', default=4,
                        help='Validation fold from split used for validation.')
    parser.add_argument('--cuda_devices', dest='cuda_devices', default='0',
                        help='Which gpus to use.')
    parser.add_argument('--state_machine', dest='state_machine', default='wait_motion',
                        help='Current state machine')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    main(args=args)
