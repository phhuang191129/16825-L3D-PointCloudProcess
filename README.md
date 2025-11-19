## Overview
In this assignment, you will implement a PointNet based architecture for classification and segmentation with point clouds (you don't need to worry about the tranformation blocks). Q1 and Q2 focus on implementing, training and testing models. Q3 asks you to quantitatively analyze model robustness. Q4 (extra point) involves locality. 

`models.py` is where you will define model structures. `train.py` loads data, trains models, logs trajectories and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain script to evaluate model accuracy and visualize segmentation result. Feel free to modify any file as needed.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Q1. Classification Model (40 points)](#q1-classification-model-40-points)
- [Q2. Segmentation Model (40 points)](#q2-segmentation-model-40-points)
- [Q3. Robustness Analysis (20 points)](#q3-robustness-analysis-20-points)
- [Q4. Bonus Question - Locality (20 points)](#q4-bonus-question---locality-20-points)

## Environment Setup
You should be able to use the environment used for previous assignments. If you need to create a new environment, please follow the set up instruction at [Assignment1](https://github.com/learning3d/assignment1). We include the same requirements.txt as A1 for your convenience.

## Data Preparation
Please download and extract the dataset for this assigment. We host the dataset on huggingface [here](https://huggingface.co/datasets/learning3dvision/assignment5/tree/main).

Download the dataset using the following commands:
```
$ sudo apt install git-lfs
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/assignment5
```

The `at_data.zip` file should be downloaded at the `assignment5` subdirecotry. You can unzip it with the command:
```
$ unzip ./assignment5/a5_data.zip -d ./
```
This will produce the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.



## Q1. Classification Model (40 points)

Run the following command, replace the `{obj_index_to_vis}` with the index of the object you intended to visualize.

```
python train.py --task cls
python eval_cls.py -i {obj_index_to_vis}
```


## Q2. Segmentation Model (40 points) 

Run the following command, replace the `{obj_index_to_vis}` with the index of the object you intended to visualize.

```
python train.py --task seg
python eval_seg.py -i {obj_index_to_vis}
```


## Q3. Robustness Analysis (20 points) 

Run the following command, replace the `{obj_index_to_vis}` with the index of the object you intended to visualize.

```
python eval_cls.py --rotate  -i {obj_index_to_vis}
python eval_seg.py --rotate -i {obj_index_to_vis}

python eval_cls.py --num_points 1000 -i {obj_index_to_vis}
python eval_seg.py --num_points 1000 -i {obj_index_to_vis}

python eval_cls.py --num_points 100 -i {obj_index_to_vis}
python eval_seg.py --num_points 100 -i {obj_index_to_vis}
```

## Q4. Bonus Question - Locality (20 points)
N/A