import re
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, brier_score_loss
from matplotlib import pyplot as plt
from matplotlib import pylab
from BasicMovementDetectionEvaluation.metrics.create_confusion_matrix import plot_confusion_matrix
from BasicMovementDetectionEvaluation.metrics.create_qq_plot import create_qq_plot
from BasicMovementDetectionEvaluation.helper.latexify import latexify


## Basic Movement detection evaluation
#
# This class contains methods to evaluate basic movement detections.
# The generated scores and plots are F1-scores, seg-scores, confusion matrix, qq plots, brier scores,
# and detection times.
#
# It is also possible to create plots for every scene containing the trajectory showing the ground truth classes for
# every position and a plot containing the predicted classes compared to the ground truth.
class BMDE(object):
    ## Convert an iterable of indices to one-hot encoded labels. (static)
    #
    # @param classes (list): contains classes names as integers
    # @param nb_classes (int): number of classes in list
    @staticmethod
    def indices_to_one_hot(classes, nb_classes):
        targets = np.array(classes).reshape(-1)
        return np.eye(nb_classes)[targets]

    ## Creates scores of basic movement detector using a test set (static)
    #
    # @param npy_path (str): path to npy file containing test results in the form of a dict:
    # ```python
    # {SCENE_NAME_0: {"x_tracked": X_POSITIONS,
    #                 "y_tracked": Y_POSITIONS",
    #                 "labels_BM0": BM0 LABELS (0 or 1),
    #                 "pred_BM0: BM0 PREDICTIONS (between 0.0 and 1.0),
    #                 ...
    #                 "pred_BMn: BMn PREDICTIONS},
    #  SCENE_NAME_1: {...}, ...}
    # ```
    # @param class_names (list(str)): list of basic movement names
    # @param output_path (str): where to save outputs, if None: npy_path is used
    # @param time_step (float): size of one time step in seconds
    # @param create_scene_outputs (bool): whether or not to create scene wise output (might take some time)
    # @param input_length (int): if trajectory is longer than predictions, the first input_length steps are skipped
    # (only necessary when create_scene_outputs is set to True)
    # @param dont_use (list(str)): list of scenes names to exclude from evaluation
    @staticmethod
    def create_scores(npy_path,
                      class_names=["turn", "straight"],
                      output_path=None,
                      time_step=0.02,
                      create_scene_outputs=True,
                      input_length=0,
                      # dont_use=[])
                      dont_use=["0090_2016_07_15_Versuch3_1_741_0"]):
        model_name = ""
        for idx, cn in enumerate(class_names):
            if idx == 0:
                model_name += cn
            else:
                model_name += " " + cn
        if output_path is None:
            output_path = os.path.join(npy_path[:npy_path.rfind("/")], "eval")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        data = np.load(npy_path, allow_pickle=True, encoding='latin1').item()
        # create input for complete evaluation
        y_test = []
        y_test_onehot = []
        predictions = []
        prob_predictions = []
        for scene_name, scene in data.items():
            if scene_name not in dont_use:
                for idx, class_name in enumerate(class_names):
                    if idx == 0:
                        concat_labels = np.expand_dims(scene['labels_' + class_name], axis=-1)
                        concat_pred = np.expand_dims(scene['pred_' + class_name], axis=-1)
                    else:
                        concat_labels = np.concatenate(
                            (concat_labels, np.expand_dims(scene['labels_' + class_name], axis=-1)), axis=-1)
                        concat_pred = np.concatenate(
                            (concat_pred, np.expand_dims(scene['pred_' + class_name], axis=-1)), axis=-1)
                for elem in concat_labels:
                    y_test.append(np.argmax(elem))
                    y_test_onehot.append(elem)
                for elem in concat_pred:
                    predictions.append(np.argmax(elem))
                    prob_predictions.append(elem)

        prob_predictions = np.array(prob_predictions)
        predictions = np.array(predictions)
        y_test = np.array(y_test)
        y_test_onehot = np.array(y_test_onehot)

        # create predictions for scene wise evaluation
        gts = []
        preds = []
        for key in list(data.keys()):
            if key not in dont_use:
                scene = data[key]
                y_scene = []
                y_scene_onehot = []
                predictions_scene = []
                prob_predictions_scene = []
                for idx, class_name in enumerate(class_names):
                    if idx == 0:
                        concat_labels = np.expand_dims(scene['labels_' + class_name], axis=-1)
                        concat_pred = np.expand_dims(scene['pred_' + class_name], axis=-1)
                    else:
                        concat_labels = np.concatenate(
                            (concat_labels, np.expand_dims(scene['labels_' + class_name], axis=-1)), axis=-1)
                        concat_pred = np.concatenate(
                            (concat_pred, np.expand_dims(scene['pred_' + class_name], axis=-1)),
                            axis=-1)
                for elem in concat_labels:
                    y_scene.append(np.argmax(elem))
                    y_scene_onehot.append(elem)
                for elem in concat_pred:
                    predictions_scene.append(np.argmax(elem))
                    prob_predictions_scene.append(elem)

                probs_scene = np.array(prob_predictions_scene)
                predictions_scene = np.array(predictions_scene)
                y_scene = np.array(y_scene)

                gts.append(y_scene)
                preds.append(predictions_scene)

                # creates plots containing predicted probabilities, ground truth, and trajectories for every scene
                if create_scene_outputs:
                    output_path_scenes = os.path.join(output_path, "trajectories")
                    colors = ["red", "green", "blue", "black", "cyan", "magenta", "orange", "yellow"]
                    # plot class probabilities
                    plt.subplots(2, 1, figsize=(8, 12))
                    plt.subplot(2, 1, 1)
                    plt.title(key)
                    plt.plot(y_scene + 1, color='purple', linewidth=3, label='gt + 1')

                    # plot probs for every class
                    for j, name in enumerate(class_names):
                        if j < 8:
                            plt.plot(probs_scene[:, j] + j, color=colors[j], linewidth=2,
                                     label='p_' + name + ' + ' + str(j))
                        else:
                            print("I can't handle more than 8 classes, because I only know 8 colors.")

                        plt.legend(loc=9, bbox_to_anchor=(1.15, 0.5))
                        lgd = pylab.legend(loc=9, bbox_to_anchor=(1.15, 0.75))
                        art = [lgd]
                        plt.grid(True)
                    plt.ylim((0.0, float(len(class_names))))

                    plt.subplot(2, 1, 2, aspect='equal')
                    # plot trajectories with classes
                    trajectory = {"x_tracked": scene["x_tracked"], "y_tracked": scene["y_tracked"]}
                    # ground truth
                    for i in range(input_length, len(trajectory['x_tracked'])):
                        idx = int(y_scene[i - input_length])
                        if idx < 8:
                            color = colors[idx]
                        else:
                            color = "black"
                            print("I can't handle more than 8 classes, because I only know 8 colors.")

                        plt.plot(trajectory['x_tracked'][i], trajectory['y_tracked'][i], color=color, marker='o')

                        # add start and end labels
                        if i == input_length:
                            plt.text(trajectory['x_tracked'][i] - 500, trajectory['y_tracked'][i] - 500, "start",
                                     color="grey")
                        if i == len(trajectory['x_tracked']) - 1:
                            plt.text(trajectory['x_tracked'][i] - 500, trajectory['y_tracked'][i] + 500, "end",
                                     color="grey")

                    # predictions
                    for i in range(input_length, len(trajectory['x_tracked'])):
                        idx = int(predictions_scene[i - input_length])
                        if idx < 8:
                            color = colors[idx]
                        else:
                            color = "black"
                            print("I can't handle more than 8 classes, because I only know 8 colors.")

                        plt.plot(trajectory['x_tracked'][i], trajectory['y_tracked'][i], color=color, marker='.')

                    if not os.path.exists(output_path_scenes):
                        os.makedirs(output_path_scenes)
                    output_file_path = os.path.join(output_path_scenes, key + ".png")
                    plt.savefig(output_file_path, additional_artists=art, bbox_inches="tight")
                    plt.clf()
                    plt.close()
                    print(key)

        # generate outputs
        # calculate f1 scores
        f1_micro = f1_score(y_test, predictions, average='micro')
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        output_string = "\nf1 micro: " + str(f1_micro) + "\nf1 macro: " + str(f1_macro) + "\nf1 weighted: " + str(
            f1_weighted) + "\n"
        print("f1 micro:", f1_micro, "f1 macro:", f1_macro, "f1 weighted:", f1_weighted)

        # calculate brier scores
        brier_scores = []
        for i in range(len(class_names)):
            bs = brier_score_loss(y_test_onehot[:, i], np.array(prob_predictions)[:, i])
            output_string += "\nbrier score " + str(class_names[i]) + ": " + str(bs)
            print("brier score ", class_names[i], bs)
            brier_scores.append(bs)

        # replace indices with class names in string
        for idx, name in enumerate(class_names):
            # remove up to 7 spaces after class name to create a table like look
            remove_spaces = len(name)
            if remove_spaces > 7:
                remove_spaces = 7
            output_string = re.sub(r'(?m)^' + str(idx) + " " * remove_spaces, name, output_string)

        with open(os.path.join(output_path, "stats.txt"), "w") as text_file:
            text_file.write(output_string)

        # create figures and save them
        latexify()
        fig_path = os.path.join(output_path, "figures")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plot_confusion_matrix(y_true=y_test, y_pred=predictions, classes=class_names, title=model_name)
        plt.savefig(os.path.join(fig_path, "conf.png"))
        plt.savefig(os.path.join(fig_path, "conf.pdf"))
        plot_confusion_matrix(y_true=y_test, y_pred=predictions, classes=class_names, title=model_name, normalize=True)
        plt.savefig(os.path.join(fig_path, "conf_norm.png"))
        plt.savefig(os.path.join(fig_path, "conf_norm.pdf"))

        # create qq plots
        for idx, c in enumerate(class_names):
            create_qq_plot(y_test_onehot[:, idx], prob_predictions[:, idx], n_bins=10, title=c)
            plt.savefig(os.path.join(fig_path, "qq_" + c + ".png"))
            plt.savefig(os.path.join(fig_path, "qq_" + c + ".pdf"))


## Example usage of BMDE
#
# This is the exmaple usage. Please do not change this. To create your own BMDE copy the code below this comment to
# your project and adjust it to your needs.
def main(args):
    class_names = ""
    if int(args.create_scene_ouput) == 1:
        create_scene_outputs = True
    else:
        create_scene_outputs = False
    # create scores
    if args.state_machine == 'wait_motion':
        class_names = ["wait", "motion"]
    elif args.state_machine == 'turn_straight':
        class_names = ["turn", "straight"]
    elif args.state_machine == 'left_right':
        class_names = ["left", "right"]
    elif args.state_machine == 'start_stop_move':
        class_names = ["start", "stop", "move"]
    if class_names:
        BMDE.create_scores(npy_path=args.npy_path,
                           class_names=class_names,
                           create_scene_outputs=create_scene_outputs)
    else:
        print('No valid state machine was passed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Density Pipeline Arguments')
    parser.add_argument('--npy_path', dest='npy_path', default='wait_motion_eval_probs.npy',
                        help='path to numpy file containing probabilities')
    parser.add_argument('--state_machine', dest='state_machine', default='wait_motion',
                        help='Current state machine')
    parser.add_argument('--create_scene_ouput', dest='create_scene_ouput', default='0',
                        help='If 1 scene wise outputs will be created (might take some time)')

    args = parser.parse_args()

    main(args=args)
