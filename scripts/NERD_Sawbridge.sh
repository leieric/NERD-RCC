#!/bin/bash

python NERDlagr_curve.py --gpus 0 1 2 --lr 1e-3 --lmbdas   -0.05 -0.10   --batch_size 1000 --latent_dim 10 --data_name "Sawbridge" --epochs 10