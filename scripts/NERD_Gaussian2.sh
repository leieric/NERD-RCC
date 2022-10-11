#!/bin/bash

python NERDlagr_curve.py --gpus 0 1 2 3 --lr 1e-3 --lmbdas  -20 -10 -6 -3 -1 -0.25   --batch_size 100 --latent_dim 100 --data_name "Gaussian2" --epochs 10