#!/bin/bash

python NERDlagr_curve.py --gpus 0 2 3 --lmbdas -0.4 -0.3 -0.2 -0.15 --batch_size 100 --data_name "MNIST" --epochs 20 --init_gan 1

# -3 -2 -1 -0.5 -0.25 -0.1 -0.05 -0.01