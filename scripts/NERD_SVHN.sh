#!/bin/bash

python NERDlagr_curve.py --gpus 0 1 3 --lmbdas -10 -2 -1 --batch_size 100 --data_name "SVHN" --epochs 30 --init_gan 1