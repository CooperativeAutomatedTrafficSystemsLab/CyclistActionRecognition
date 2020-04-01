# Under Cosntruction
This repository is still under construction. Dataset and publication associated to the repository will be made public
as soon as possible.

# Cyclist Action Recognition
This repository contains code to train, validate, and test networks for cyclist action recognition based on image 
sequences, optical flow sequences, and trajectories.
It was used to produce the results from *Image Sequence Based Cyclist Action Recognition Using Multi-Stream 3D 
Convolution* (link will be added soon).

## Motion Sequence Based Network

### Prepare Dataset

The dataset, including a short description of the dataset, can be found here (link will be added as soon as dataset is 
available). Since the dataset is rather larger (~63 GB), we need to create a numpy memory map first, so we don't need to
load the complete dataset to memory (if you have enough memory, you can also load the complete dataset into your 
memory).

Download the dataset and extract it to a folder called dataset (e.g.: cyclist_action_recognition/dataset). Copy the
split.json to the same folder. Make sure you have enough space on your hard drive. The dataset is compressed using 7zip.
Extract it by (might take a while):

```bash
cyclist_action_recognition/dataset$ 7z e of_smlr.7z
```

and run:
```bash
python3 ImageSequenceDataset/prepare_memmap/create_np_memmap.py \
--dataset_path PATH_TO_DATASET/cyclist_action_recognition/dataset
```
The dataset will be prepared as two numpy memory maps containing the network inputs for training and validation.

### Performing a Grid Search

To perform a grid search, start *pipeline_motion_sequences/gridsearch_motion_sequences.py*:

```bash
python3 pipeline_motion_sequences/gridsearch_motion_sequences.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition/ \
--json_path ./sweeps/wait_motion.json
```

The grid search will train a network for every configuration in *wait_motion.json* and add train and validation scores
to tensorboard. Start tensorboard by:

```bash
tensorboard --logdir=PATH_TO_DATASET/cyclist_action_recognition/trained_models
```

### Train Final Estimator

To train the final estimator run:

```bash
python3 pipeline_motion_sequences/train_final_estimator.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition/ \
--json_path ./final/wait_motion_final.json \
--steps NUMBER_OF_STEPS_TO_TRAIN
```

The estimator will be trained until step NUMBER_OF_STEPS_TO_TRAIN and the final checkpoint will be saved. This is the 
checkpoint used to create the test probabilities.

### Create Test Probabilities

To create test probabilities run:

```bash
python3 pipeline_motion_sequences/create_test_probs.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition/ \
--state_machine NAME_OF_STATE_MACHINE 
```

The checkpoint file must be placed in 
*PATH_TO_DATASET/cyclist_action_recognition/trained_models/final/NAME_OF_STATE_MACHINE*. An output file *probs.npy* will
be saved to the same folder.

## Trajectory Based Network

The dataset, including a short description of the dataset, can be found here (link will be added as soon as dataset is 
available). Compared to the of dataset, the dataset is small, so we can load it directly to RAM.

Download the dataset and extract it to a folder called dataset (e.g.: cyclist_action_recognition_trajectory/dataset). 
Copy the split.json to the same folder.

### Performing a Grid Search

To perform a grid search, start *pipeline_trajectory_based/gridsearch_trajectory_based.py*:

```bash
python3 pipeline_trajectory_based/gridsearch_trajectory_based.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition_trajectory/ \
--json_path ./sweeps/wait_motion.json
```

The grid search will train a network for every configuration in *wait_motion.json* and add train and validation scores
to tensorboard. Start tensorboard by:

```bash
tensorboard --logdir=PATH_TO_DATASET/cyclist_action_recognition_trajectory/trained_models
```

### Train Final Estimator

To train the final estimator run:

```bash
python3 pipeline_trajectory_based/train_final_estimator_trajectory_based.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition_trajectory/ \
--json_path ./final/wait_motion_final.json \
--steps NUMBER_OF_STEPS_TO_TRAIN
```

The estimator will be trained until step NUMBER_OF_STEPS_TO_TRAIN and the final checkpoint will be saved. This is the 
checkpoint used to create the test probabilities.

### Create Test Probabilities

To create test probabilities run:

```bash
python3 pipeline_trajectory_based/create_test_probs.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition_trajectory/ \
--state_machine NAME_OF_STATE_MACHINE 
```

The checkpoint file must be placed in 
*PATH_TO_DATASET/cyclist_action_recognition_trajectory/trained_models/final/NAME_OF_STATE_MACHINE*. An output file 
*probs.npy* will be saved to the same folder.

## Evaluate Results

### Prepare Probabilities 

Prepare proabilities for evaluation script:

```bash
python3 BasicMovementDetectionEvaluation/create_evaluation_inputs.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition_trajectory/ \
--state_machine NAME_OF_STATE_MACHINE 
```

### Create Evaluation

To create evaluation results run:

```bash
python3 BasicMovementDetectionEvaluation/create_evaluation_inputs.py \
--networkPath PATH_TO_DATASET/cyclist_action_recognition_trajectory/final/wait_motion/wait_motion_eval_probs.npy \
--state_machine NAME_OF_STATE_MACHINE 
```

The evaluation results will be created in the same folder. The script generate F1-scores, Brier-scores, confusion 
matrices, and Q-Q plots.
