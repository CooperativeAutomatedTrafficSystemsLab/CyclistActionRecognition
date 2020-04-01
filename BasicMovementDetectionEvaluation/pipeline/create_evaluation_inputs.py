import argparse
import os
import numpy as np


## Creates inputs for evaluation script
# network_path should point to the path where trained_models/final/NAME_STATE_MACHINE is located
# outputs are written to trained_models/final/NAME_STATE_MACHINE/NAME_STATE_MACHINE_eval_probs.npy
#
def main(args):
    # set path to state machine
    network_path = args.network_path
    estimator_name = args.state_machine
    estimator_paths = {"wait_motion": os.path.join(network_path, "trained_models/final/wait_motion"),
                       "turn_straight": os.path.join(network_path, "trained_models/final/turn_straight"),
                       "left_right": os.path.join(network_path, "trained_models/final/left_right"),
                       "start_stop_move": os.path.join(network_path, "trained_models/final/start_stop_move")}
    output_path = os.path.join(network_path, estimator_paths[estimator_name], estimator_name + "_eval_probs.npy")

    output_dict = {}
    # load probabilities
    probs_path = os.path.join(estimator_paths[estimator_name], 'probs.npy')
    probs = np.load(probs_path, allow_pickle=True, encoding='latin1').item()
    # convert all scenes to needed format
    for scene_name in list(probs.keys()):
        x_tracked = []
        y_tracked = []
        z_tracked = []
        x_smoothed = []
        y_smoothed = []
        z_smoothed = []
        ts = []
        pred_wait = []
        labels_wait = []
        pred_motion = []
        labels_motion = []
        pred_turn = []
        labels_turn = []
        pred_straight = []
        labels_straight = []
        pred_left = []
        labels_left = []
        pred_right = []
        labels_right = []
        pred_start = []
        labels_start = []
        pred_stop = []
        labels_stop = []
        pred_move = []
        labels_move = []
        appended = False
        scene_counter = 0
        if estimator_name == "wait_motion":
            for i in range(len(probs[scene_name]["pred_wait_motion"])):
                if np.sum(probs[scene_name]["labels_wait_motion"][i]) == 1:
                    x_tracked.append(probs[scene_name]["x_tracked"][i + 49])
                    y_tracked.append(probs[scene_name]["y_tracked"][i + 49])
                    z_tracked.append(probs[scene_name]["z_tracked"][i + 49])
                    x_smoothed.append(probs[scene_name]["x_smoothed"][i + 49])
                    y_smoothed.append(probs[scene_name]["y_smoothed"][i + 49])
                    z_smoothed.append(probs[scene_name]["z_smoothed"][i + 49])
                    ts.append(probs[scene_name]["ts"][i + 49])
                    pred_wait.append(probs[scene_name]["pred_wait_motion"][i][0])
                    labels_wait.append(probs[scene_name]["labels_wait_motion"][i][0])
                    pred_motion.append(probs[scene_name]["pred_wait_motion"][i][1])
                    labels_motion.append(probs[scene_name]["labels_wait_motion"][i][1])
                    appended = True
                elif appended:
                    output_dict.update({scene_name + "_" + str(scene_counter):
                                            {"x_tracked": np.array(x_tracked),
                                             "y_tracked": np.array(y_tracked),
                                             "z_tracked": np.array(z_tracked),
                                             "x_smoothed": np.array(x_smoothed),
                                             "y_smoothed": np.array(y_smoothed),
                                             "z_smoothed": np.array(z_smoothed),
                                             "ts": np.array(ts),
                                             "pred_wait": np.array(pred_wait),
                                             "labels_wait": np.array(labels_wait),
                                             "pred_motion": np.array(pred_motion),
                                             "labels_motion": np.array(labels_motion),
                                             }})
                    x_tracked = []
                    y_tracked = []
                    z_tracked = []
                    x_smoothed = []
                    y_smoothed = []
                    z_smoothed = []
                    ts = []
                    pred_wait = []
                    labels_wait = []
                    pred_motion = []
                    labels_motion = []
                    appended = False
                    scene_counter += 1

            if appended:
                output_dict.update({scene_name + "_" + str(scene_counter):
                                        {"x_tracked": np.array(x_tracked),
                                         "y_tracked": np.array(y_tracked),
                                         "z_tracked": np.array(z_tracked),
                                         "x_smoothed": np.array(x_smoothed),
                                         "y_smoothed": np.array(y_smoothed),
                                         "z_smoothed": np.array(z_smoothed),
                                         "ts": np.array(ts),
                                         "pred_wait": np.array(pred_wait),
                                         "labels_wait": np.array(labels_wait),
                                         "pred_motion": np.array(pred_motion),
                                         "labels_motion": np.array(labels_motion),
                                         }})
        elif estimator_name == "turn_straight":
            for i in range(len(probs[scene_name]["pred_turn_straight"])):
                if np.sum(probs[scene_name]["labels_turn_straight"][i]) == 1:
                    x_tracked.append(probs[scene_name]["x_tracked"][i + 49])
                    y_tracked.append(probs[scene_name]["y_tracked"][i + 49])
                    z_tracked.append(probs[scene_name]["z_tracked"][i + 49])
                    x_smoothed.append(probs[scene_name]["x_smoothed"][i + 49])
                    y_smoothed.append(probs[scene_name]["y_smoothed"][i + 49])
                    z_smoothed.append(probs[scene_name]["z_smoothed"][i + 49])
                    ts.append(probs[scene_name]["ts"][i + 49])
                    pred_turn.append(probs[scene_name]["pred_turn_straight"][i][0])
                    labels_turn.append(probs[scene_name]["labels_turn_straight"][i][0])
                    pred_straight.append(probs[scene_name]["pred_turn_straight"][i][1])
                    labels_straight.append(probs[scene_name]["labels_turn_straight"][i][1])
                    appended = True
                elif appended:
                    output_dict.update({scene_name + "_" + str(scene_counter):
                                            {"x_tracked": np.array(x_tracked),
                                             "y_tracked": np.array(y_tracked),
                                             "z_tracked": np.array(z_tracked),
                                             "x_smoothed": np.array(x_smoothed),
                                             "y_smoothed": np.array(y_smoothed),
                                             "z_smoothed": np.array(z_smoothed),
                                             "ts": np.array(ts),
                                             "pred_turn": np.array(pred_turn),
                                             "labels_turn": np.array(labels_turn),
                                             "pred_straight": np.array(pred_straight),
                                             "labels_straight": np.array(labels_straight),
                                             }})
                    x_tracked = []
                    y_tracked = []
                    z_tracked = []
                    x_smoothed = []
                    y_smoothed = []
                    z_smoothed = []
                    ts = []
                    pred_turn = []
                    labels_turn = []
                    pred_straight = []
                    labels_straight = []
                    appended = False
                    scene_counter += 1
            if appended:
                output_dict.update({scene_name + "_" + str(scene_counter):
                                        {"x_tracked": np.array(x_tracked),
                                         "y_tracked": np.array(y_tracked),
                                         "z_tracked": np.array(z_tracked),
                                         "x_smoothed": np.array(x_smoothed),
                                         "y_smoothed": np.array(y_smoothed),
                                         "z_smoothed": np.array(z_smoothed),
                                         "ts": np.array(ts),
                                         "pred_turn": np.array(pred_turn),
                                         "labels_turn": np.array(labels_turn),
                                         "pred_straight": np.array(pred_straight),
                                         "labels_straight": np.array(labels_straight),
                                         }})
        elif estimator_name == "left_right":
            for i in range(len(probs[scene_name]["pred_left_right"])):
                if np.sum(probs[scene_name]["labels_left_right"][i]) == 1:
                    x_tracked.append(probs[scene_name]["x_tracked"][i + 49])
                    y_tracked.append(probs[scene_name]["y_tracked"][i + 49])
                    z_tracked.append(probs[scene_name]["z_tracked"][i + 49])
                    x_smoothed.append(probs[scene_name]["x_smoothed"][i + 49])
                    y_smoothed.append(probs[scene_name]["y_smoothed"][i + 49])
                    z_smoothed.append(probs[scene_name]["z_smoothed"][i + 49])
                    ts.append(probs[scene_name]["ts"][i + 49])
                    pred_left.append(probs[scene_name]["pred_left_right"][i][0])
                    labels_left.append(probs[scene_name]["labels_left_right"][i][0])
                    pred_right.append(probs[scene_name]["pred_left_right"][i][1])
                    labels_right.append(probs[scene_name]["labels_left_right"][i][1])
                    appended = True
                elif appended:
                    output_dict.update({scene_name + "_" + str(scene_counter):
                                            {"x_tracked": np.array(x_tracked),
                                             "y_tracked": np.array(y_tracked),
                                             "z_tracked": np.array(z_tracked),
                                             "x_smoothed": np.array(x_smoothed),
                                             "y_smoothed": np.array(y_smoothed),
                                             "z_smoothed": np.array(z_smoothed),
                                             "ts": np.array(ts),
                                             "pred_left": np.array(pred_left),
                                             "labels_left": np.array(labels_left),
                                             "pred_right": np.array(pred_right),
                                             "labels_right": np.array(labels_right),
                                             }})
                    x_tracked = []
                    y_tracked = []
                    z_tracked = []
                    x_smoothed = []
                    y_smoothed = []
                    z_smoothed = []
                    ts = []
                    pred_left = []
                    labels_left = []
                    pred_right = []
                    labels_right = []
                    appended = False
                    scene_counter += 1
            if appended:
                output_dict.update({scene_name + "_" + str(scene_counter):
                                        {"x_tracked": np.array(x_tracked),
                                         "y_tracked": np.array(y_tracked),
                                         "z_tracked": np.array(z_tracked),
                                         "x_smoothed": np.array(x_smoothed),
                                         "y_smoothed": np.array(y_smoothed),
                                         "z_smoothed": np.array(z_smoothed),
                                         "ts": np.array(ts),
                                         "pred_left": np.array(pred_left),
                                         "labels_left": np.array(labels_left),
                                         "pred_right": np.array(pred_right),
                                         "labels_right": np.array(labels_right),
                                         }})
        elif estimator_name == "start_stop_move":
            for i in range(len(probs[scene_name]["pred_start_stop_move"])):
                if np.sum(probs[scene_name]["labels_start_stop_move"][i]) == 1:
                    x_tracked.append(probs[scene_name]["x_tracked"][i + 49])
                    y_tracked.append(probs[scene_name]["y_tracked"][i + 49])
                    z_tracked.append(probs[scene_name]["z_tracked"][i + 49])
                    x_smoothed.append(probs[scene_name]["x_smoothed"][i + 49])
                    y_smoothed.append(probs[scene_name]["y_smoothed"][i + 49])
                    z_smoothed.append(probs[scene_name]["z_smoothed"][i + 49])
                    ts.append(probs[scene_name]["ts"][i + 49])
                    pred_start.append(probs[scene_name]["pred_start_stop_move"][i][0])
                    labels_start.append(probs[scene_name]["labels_start_stop_move"][i][0])
                    pred_stop.append(probs[scene_name]["pred_start_stop_move"][i][1])
                    labels_stop.append(probs[scene_name]["labels_start_stop_move"][i][1])
                    pred_move.append(probs[scene_name]["pred_start_stop_move"][i][2])
                    labels_move.append(probs[scene_name]["labels_start_stop_move"][i][2])
                    appended = True
                elif appended:
                    output_dict.update({scene_name + "_" + str(scene_counter):
                                            {"x_tracked": np.array(x_tracked),
                                             "y_tracked": np.array(y_tracked),
                                             "z_tracked": np.array(z_tracked),
                                             "x_smoothed": np.array(x_smoothed),
                                             "y_smoothed": np.array(y_smoothed),
                                             "z_smoothed": np.array(z_smoothed),
                                             "ts": np.array(ts),
                                             "pred_start": np.array(pred_start),
                                             "labels_start": np.array(labels_start),
                                             "pred_stop": np.array(pred_stop),
                                             "labels_stop": np.array(labels_stop),
                                             "pred_move": np.array(pred_move),
                                             "labels_move": np.array(labels_move),
                                             }})
                    x_tracked = []
                    y_tracked = []
                    z_tracked = []
                    x_smoothed = []
                    y_smoothed = []
                    z_smoothed = []
                    ts = []
                    pred_start = []
                    labels_start = []
                    pred_stop = []
                    labels_stop = []
                    pred_move = []
                    labels_move = []
                    appended = False
                    scene_counter += 1
            if appended:
                output_dict.update({scene_name + "_" + str(scene_counter):
                                        {"x_tracked": np.array(x_tracked),
                                         "y_tracked": np.array(y_tracked),
                                         "z_tracked": np.array(z_tracked),
                                         "x_smoothed": np.array(x_smoothed),
                                         "y_smoothed": np.array(y_smoothed),
                                         "z_smoothed": np.array(z_smoothed),
                                         "ts": np.array(ts),
                                         "pred_start": np.array(pred_start),
                                         "labels_start": np.array(labels_start),
                                         "pred_stop": np.array(pred_stop),
                                         "labels_stop": np.array(labels_stop),
                                         "pred_move": np.array(pred_move),
                                         "labels_move": np.array(labels_move),
                                         }})
    np.save(output_path, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--networkPath', dest='network_path',
                        default='cyclist_action_recognition_trajectory/',
                        help='Path to trained network used to create test results.')
    parser.add_argument('--state_machine', dest='state_machine', default='wait_motion',
                        help='Current state machine')

    args = parser.parse_args()
    main(args=args)
