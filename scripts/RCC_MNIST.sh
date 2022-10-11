#!/bin/bash
# python one_shot_PFRlagr.py --gpu 1 --lmbdas -1 --data_name "SVHN" --N 30000
python one_shot_PFRlagr.py --gpu 0 --lmbdas -10 -2 -1 -0.5 -0.4 -0.3 -0.25 -0.2 -0.15 -0.1 -0.05 --data_name "MNIST" --N 30000