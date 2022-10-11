#!/bin/bash

python NERDlagr_curve.py --gpus 0 --lmbdas 10 -2 -1 -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 -0.01 --batch_size 100 --data_name "SVHN" --epochs 30 --init_gan 1