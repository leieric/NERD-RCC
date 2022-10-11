#!/bin/bash

python NERDlagr_curve.py --gpus 0 --lr 1e-3 --lmbdas  -0.1 -0.05 -0.01   --batch_size 100 --latent_dim 100 --data_name "Gaussian" --epochs 10