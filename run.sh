#!/bin/bash

# Q1
python train.py --task cls
python eval_cls.py


# Q2
python train.py --task seg
python eval_seg.py

# Q3 
python eval_cls.py --rotate
python eval_seg.py --rotate

python eval_cls.py --num_points 1000
python eval_seg.py --num_points 1000

python eval_cls.py --num_points 100
python eval_seg.py --num_points 100