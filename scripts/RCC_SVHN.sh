#!/bin/bash
python one_shot_PFRlagr.py --gpu 1 --lmbdas -1 --data_name "SVHN" --N 30000
#python one_shot_PFRlagr.py --gpu 1 --lmbdas -10 -2 -1 -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 -0.01 --data_name "SVHN" --N 30000