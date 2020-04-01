import numpy as np
import argparse
from .EgoTransformation.EgoTransformation import Ego, crop_trajectory, crop_orientation
from tqdm import tqdm
import cv2 as cv


class PreprocessTrajectoryDataset(object):
    ##
    # @param split: (string) path to split.json file
    # @param n_in:  (int) length of input trajectory
    # @param n_out: (int) length of output (gt) trajectory
    # @param scale: (float) scale for the word coordinates (they are represented in mm by default and scaled into m with scale=0.001)
    # @param values_to_be_stored_input: (list) list of names stored in input trajectory, by default only tracked x and y positions are used
    # @param values_to_be_stored_output: (list) list of names stored in output trajectory, by default only smoothed x and y positions are used
    # @param ts_used_for_input: (list) indices of timestemps to be used in input trajectory, by default all timestemps are used
    # @param ts_used_for_gt: (list) indices of timestemps to be used in output trajectory, by default all timestemps are used
    # @param use_orientation_from_dataset: (bool) decides if the orientation is computed at runtime (vanilla transformation) or looked up from dataset (maartens transformation)
    def __init__(self, split, n_in, n_out=None, scale=0.001,
                 values_to_be_stored_input=['x_tracked', 'y_tracked'],
                 values_to_be_stored_output=['x_smoothed', 'y_smoothed'],
                 ts_used_for_input=None,
                 ts_used_for_gt=None,
                 use_orientation_from_dataset=True):

        self.n_in = n_in
        self.n_out = n_out
        if isinstance(split, str):
            import json
            self.split = json.load(open(split, "rb"))
        elif isinstance(split, dict):
            self.split = split

        # initialize ego transformation
        self.ego = Ego(scale=scale)

        self.values_to_be_stored_input = values_to_be_stored_input
        self.values_to_be_stored_output = values_to_be_stored_output
        self.use_orientation_from_dataset = use_orientation_from_dataset

        if isinstance(ts_used_for_input, list):
            ts_used_for_input = filter(lambda x: x < self.n_in, ts_used_for_input)
            if len(ts_used_for_input) < 2:
                raise ValueError('You can only specify timesteps < n_in for the input trajectory')

        if isinstance(ts_used_for_gt, list) and not isinstance(n_out, type(None)):
            ts_used_for_gt = filter(lambda x: x < self.n_out, ts_used_for_gt)
            if len(ts_used_for_gt) < 2:
                raise ValueError('You can only specify timesteps < n_out for the output trajectory')
        else:
            ts_used_for_gt = None

        self.ts_used_for_input = ts_used_for_input
        self.ts_used_for_gt = ts_used_for_gt
        self.pos_list = ['x', 'y', 'z', 'x_tracked', 'y_tracked', 'z_tracked', 'x_smoothed',
                         'y_smoothed']

    ## splits the Dataset based on the split.json into train and testdata
    # @param D: (dict) Dataset. A dict witch contains positions and labels. Is addressed by sequence name as key
    # @return D_train: (dict) Train dataset. A dict witch contains positions and labels. Is adressed by sequence name as key
    # @return D_test: (dict) Test dataset. A dict witch contains positions and labels. Is adressed by sequence name as key
    # @return Validation_Fold: (dict). A dict witch contains the validation fold for the trainset. Is addressed by sequence name as key
    def split_train_test(self, D):
        D_test = {}
        D_train = {}
        Validation_Fold = {}
        for key, value in D.items():
            if key in self.split.keys():
                if self.split[str(key)] == "test":
                    D_test.update({key: value})
                else:
                    D_train.update({key: value})
                    Validation_Fold.update({key: int(self.split[key].split("train_")[-1])})
        return D_train, D_test, Validation_Fold

    def get_ms_from_set(self, D, name):
        # get mss from dataset
        ms_hk1 = D[name]['ms_hk1']
        ms_hk2 = D[name]['ms_hk2']

        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 80]
        _, dummy = cv.imencode('.jpg', np.zeros((192, 192, 3), dtype=np.uint8), encode_param)

        # crop ms
        ms_hk1 = ms_hk1[self.n_in - 1:]
        ms_hk2 = ms_hk2[self.n_in - 1:]

        for i in range(len(ms_hk1)):
            if type(ms_hk1[i]) is not list:
                print("ms missing hk1", name, i)
                ms_hk1[i] = dummy

        for i in range(len(ms_hk2)):
            if type(ms_hk2[i]) is not list:
                print("ms missing hk2", name, i)
                ms_hk2[i] = dummy

        return ms_hk1, ms_hk2

    def get_of_from_set(self, D, name):
        # get ofs from dataset
        of_hk1 = D[name]['of_hk1']
        of_hk2 = D[name]['of_hk2']

        dummy = 0

        # crop of
        of_hk1 = of_hk1[self.n_in - 1:]
        of_hk2 = of_hk2[self.n_in - 1:]

        for i in range(len(of_hk1)):
            if type(of_hk1[i]) is not bytes:
                print("of missing hk1", name, i)
                of_hk1[i] = dummy

        for i in range(len(of_hk2)):
            if type(of_hk2[i]) is not bytes:
                print("of missing hk2", name, i)
                of_hk2[i] = dummy

        return of_hk1, of_hk2

    def get_mhis_from_set(self, D, name):
        # get mhis from dataset
        mhi_hk1 = D[name]['mhi_hk1']
        mhi_hk2 = D[name]['mhi_hk2']

        import pickle  # cPickle as pickle
        import zlib

        dummy = zlib.compress(pickle.dumps(np.zeros((192, 192, 2), dtype=np.uint8)))

        # crop mhis
        mhi_hk1 = mhi_hk1[self.n_in - 1:]
        mhi_hk2 = mhi_hk2[self.n_in - 1:]

        for i in range(len(mhi_hk1)):
            if type(mhi_hk1[i]) is not str:
                print("mhi missing hk1", name, i)
                mhi_hk1[i] = dummy

        for i in range(len(mhi_hk2)):
            if type(mhi_hk2[i]) is not str:
                print("mhi missing hk2", name, i)
                mhi_hk2[i] = dummy

        return mhi_hk1, mhi_hk2

    def get_labels_from_set(self, D, name):
        output = []
        for i in range(len(D[name]['x_tracked'])):
            output.append([D[name][value][i] for value in self.values_to_be_stored_output])

        # crop mhis
        output = output[self.n_in - 1:]

        return output

    ## returns all position samples from one sequence
    # The time horizon is cropped by a sliding window over the timesteps!
    #       X
    #
    #   |--------------|-----------|...........
    #   (t=0)-n_in     (t=0)       (t=0)+n_out
    #
    #   .|--------------|-----------|..........
    #    (t=1)-n_in     (t=1)       (t=1)+n_out
    #
    #   ..|--------------|-----------|.........
    #     (t=2)-n_in     (t=2)       (t=2)+n_out
    #
    #
    # @param D: (dict) Dataset. A dict witch contains positions and labels. Is addressed by sequence name as key
    # @param name: (string) key to address a sequence
    # @return input trajectory (np.array)  Input trajectory positions all timesteps are stacked in one array
    # @return output trajectory (np.array, only if self.n_out is not none) Output trajectory positions all timesteps are stacked in one array
    # @return Validation_Fold: (int). Number of the validation fold
    # @return orientation: (np.array).
    def get_trajectory_from_set(self, D, name):
        name_without_part = name.split("_part_")[0]
        if name_without_part in self.split.keys() and self.split[name_without_part] != 'test':
            Validation_Fold = int(self.split[name_without_part].split("train_")[-1])
        else:
            Validation_Fold = None

        if self.use_orientation_from_dataset:
            if 'orientation' in D[name].keys():
                orientation = crop_orientation(np.array(D[name]['orientation']), n_in=self.n_in, n_out=self.n_out)
                orientation = orientation - np.pi * 0.5
            else:
                orientation = None
                print("WARNING: use_orientation_from_dataset is set to True, but no orientation information was found!")
        else:
            orientation = None
        X = np.stack([D[name][value] for value in self.values_to_be_stored_input if value in self.pos_list], axis=-1)
        if self.n_out:
            y = np.stack([D[name][value] for value in self.values_to_be_stored_output if value in self.pos_list],
                         axis=-1)
            X_world_tracked, _ = crop_trajectory(X, n_in=self.n_in, n_out=self.n_out, dims=X.shape[1])
            _, y_world_smooth = crop_trajectory(y, n_in=self.n_in, n_out=self.n_out)
            return np.squeeze(X_world_tracked[..., self.ts_used_for_input, :]), \
                   np.squeeze(y_world_smooth[..., self.ts_used_for_gt, :]), \
                   Validation_Fold, orientation
        else:
            X_world_tracked = crop_trajectory(X, n_in=self.n_in, n_out=self.n_out, dims=X.shape[1])
            return np.squeeze(X_world_tracked[..., self.ts_used_for_input, :]), Validation_Fold, orientation

    ## returns all input labels samples from one sequence
    # The timehorizon is cropped by a sliding window over the timesteps!
    #       X
    #
    #   |--------------|...........
    #   (t=0)-n_in     (t=0)
    #
    #   .|--------------|..........
    #    (t=1)-n_in     (t=1)
    #
    #   ..|--------------|.........
    #     (t=2)-n_in     (t=2)
    #
    # @param D: (dict) Dataset. A dict witch contains positions and labels. Is addressed by sequence name as key
    # @param name: (string) key to address a sequence
    # @return labels (np.array)  labels for input timehorizon at timesteps, stacked in one array
    def get_input_labels_from_set(self, D, name):
        # only get labels, not positions (saved in self.pos_list)
        values = [D[name][value] for value in self.values_to_be_stored_input if value not in self.pos_list]
        if len(values) > 0:
            labels = np.stack(values, axis=-1)
            if self.n_out:
                labels, _ = crop_trajectory(labels, n_in=self.n_in, n_out=self.n_out, dims=labels.shape[-1])
            else:
                labels = crop_trajectory(labels, n_in=self.n_in, n_out=self.n_out, dims=labels.shape[-1])
            if not isinstance(self.ts_used_for_input, type(None)):
                return np.squeeze(labels[..., self.ts_used_for_input, :])
            else:
                return np.squeeze(labels)
        return None

    ## returns all position samples from one sequence
    # The time horizon is cropped by a sliding window over the timesteps!
    #       X
    #
    #   |-----------|...........
    #   (t=0)       (t=0)+n_out
    #
    #   .|-----------|..........
    #    (t=1)       (t=1)+n_out
    #
    #   ..|-----------|.........
    #    (t=2)       (t=2)+n_out
    #
    #
    # @param D: (dict) Dataset. A dict witch contains positions and labels. Is addressed by sequence name as key
    # @param name: (string) key to address a sequence
    # @return labels (np.array)  labels for output time horizon at timesteps, stacked in one array
    def get_output_labels_from_set(self, D, name):
        # only get labels, not positions (saved in self.pos_list)
        values = [D[name][value] for value in self.values_to_be_stored_output if value not in self.pos_list]
        if len(values) > 0:
            labels = np.stack(values, axis=-1)
            if self.n_out:
                _, labels = crop_trajectory(labels, n_in=self.n_in, n_out=self.n_out, dims=labels.shape[-1])
            else:
                labels = crop_trajectory(labels, n_in=self.n_in, n_out=self.n_out, dims=labels.shape[-1])
            if not isinstance(self.ts_used_for_gt, type(None)):
                return np.squeeze(labels[..., self.ts_used_for_gt, :])
            else:
                return np.squeeze(labels)
        return None

    ## transforms trajectories, which are stacked in one array from world coordinates to ego coordinates
    # calls the ego transform
    # @param X_world: (np.array) input trajectories to be transformed
    # @param y_world: (np.array) input trajectories to be transformed (optional)
    # @param Validation_Fold:
    # @param orientation:
    # @return:
    def transform(self, X_world, y_world=None, Validation_Fold=None, orientation=None):
        return self.ego.transform(X_world, y_world, orientation), [Validation_Fold] * X_world.shape[0]

    def invers_transform(self, X_ego, y_ego=None):
        return self.ego.inverse_transform(X_ego, y_ego)

    def get_transform_parameters(self):
        return self.ego.reference_position, self.ego.rotation_angle, self.ego.scale

    def transform_set(self, D, use_mhis=False, use_ms=False, use_of=False, use_n_first=None):
        X = []
        y = []
        Validation_Fold = []
        for i, name in enumerate(tqdm(D.keys())):
            if use_n_first is not None:
                if i > use_n_first:
                    break
            if use_mhis or use_ms or use_of or True:
                y_labels = self.get_labels_from_set(D, name)
                X_labels = None
            else:
                X_labels = self.get_input_labels_from_set(D, name=name)
                y_labels = self.get_output_labels_from_set(D, name=name)
            if self.n_out:
                X_world_tracked, y_world_smooth, VF, orientation = self.get_trajectory_from_set(D, name=name)
                # handels the case if the traj is exactly one frame long
                if len(X_world_tracked.shape) == 2:
                    X_world_tracked = np.expand_dims(X_world_tracked, axis=0)
                    y_world_smooth = np.expand_dims(y_world_smooth, axis=0)
                    if not isinstance(X_labels, type(None)):
                        X_labels = np.expand_dims(X_labels, axis=0)
                    if not isinstance(y_labels, type(None)):
                        y_labels = np.expand_dims(y_labels, axis=0)

                (X_ego, y_ego), VF = self.transform(X_world_tracked, y_world_smooth,
                                                    Validation_Fold=VF, orientation=orientation)
                Validation_Fold.append(VF)

            else:
                X_world_tracked, VF, orientation = self.get_trajectory_from_set(D, name=name)
                if use_mhis:
                    mhi_hk1, mhi_hk2 = self.get_mhis_from_set(D, name=name)
                if use_ms:
                    ms_hk1, ms_hk2 = self.get_ms_from_set(D, name=name)
                if use_of:
                    of_hk1, of_hk2 = self.get_of_from_set(D, name=name)
                (X_ego), VF = self.transform(X_world_tracked, Validation_Fold=VF, orientation=orientation)
                Validation_Fold.append(VF)

            if not isinstance(X_labels, type(None)):
                if X_ego.shape[0:-1] == X_labels.shape[0:-1]:
                    Xt = np.concatenate([X_ego, X_labels], axis=-1)
                    X.append(Xt)
            else:
                if use_mhis:
                    for i in range(len(mhi_hk1)):
                        sample = {"img_hk1": mhi_hk1[i],
                                  "img_hk2": mhi_hk2[i],
                                  "trajectory": X_ego[i]}
                        X.append(sample)
                elif use_ms and not use_of:
                    for i in range(len(ms_hk1)):
                        sample = {"ms_hk1": ms_hk1[i],
                                  "ms_hk2": ms_hk2[i],
                                  "trajectory": X_ego[i]}
                        X.append(sample)
                elif use_of and not use_ms:
                    for i in range(len(of_hk1)):
                        sample = {"of_hk1": of_hk1[i],
                                  "of_hk2": of_hk2[i],
                                  "trajectory": X_ego[i]}
                        X.append(sample)
                elif use_ms and use_of:
                    for i in range(len(ms_hk1)):
                        sample = {"ms_hk1": ms_hk1[i],
                                  "ms_hk2": ms_hk2[i],
                                  "of_hk1": of_hk1[i],
                                  "of_hk2": of_hk2[i],
                                  "trajectory": X_ego[i]}
                        X.append(sample)
                else:
                    Xt = X_ego
                    X.append(Xt)

            if not isinstance(y_labels,
                              type(None)) and self.n_out is not None:
                if y_ego.shape[0:-1] == y_labels.shape[0:-1]:
                    yt = np.concatenate([y_ego, y_labels], axis=-1)
                    y.append(yt)
            elif y_labels is not None:
                y.extend(y_labels)
            else:
                y.append(y_ego)

        if self.n_out:
            if len(X) > 0:
                return np.concatenate(X, axis=0), \
                       np.concatenate(y, axis=0), \
                       np.concatenate(Validation_Fold, axis=0)
            else:
                return None, None, None
        elif y:
            if len(X) > 0:
                return X, y, np.concatenate(Validation_Fold, axis=0)
            else:
                return None, None, None
        else:
            if len(X) > 0:
                return np.concatenate(X, axis=0), np.concatenate(Validation_Fold, axis=0)
            else:
                return None, None

    def filter_set_by_label(self, D, label):
        if label == 'complete':
            return D
        D_new = {}
        for name, trajectory in D.iteritems():
            index_list = [idx for idx, label_active in enumerate(trajectory[label]) if label_active == 1]
            if index_list:
                # get monotonic index
                m = [index_list[0]]
                for i in range(1, len(index_list) - 1):
                    if not index_list[i - 1] == index_list[i] - 1:
                        m.append(index_list[i - 1])
                m.append(index_list[-1])

                # get indices for label start and stop
                m_start = m[:-1]
                m_stop = m[1:]
                # subtract input horizon from start label indices
                m_start = map(lambda x: max(x - self.n_in, 0), m_start)
                for i in range(len(m_start)):
                    if self.n_out:
                        if (m_stop[i] - m_start[i]) < (self.n_in + self.n_out):
                            continue
                    else:
                        if (m_stop[i] - m_start[i]) < self.n_in:
                            continue
                    mps_dict = {}
                    for key in trajectory.keys():
                        mps_dict.update({key: trajectory[key][m_start[i]:m_stop[i]]})
                    D_new.update({name + '_part_' + str(i + 1): mps_dict})
        return D_new

    ## returns the number of sequences witch include the specified label
    #
    def get_num_labels(self, D, label):
        return sum([any(D[str(key)][label]) for key in D.keys()])
