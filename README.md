# NERD-RCC
Neural estimation of the rate-distortion function and its applications to reverse channel coding and compression. Provides a neural estimator (NERD) for estimating the rate-distortion function R(D) from i.i.d. samples. Uses NERD to implement single-shot lossy compression with guarantees on achievable rate-distortion.


For full details, see:

Eric Lei, Hamed Hassani, and Shirin Saeedi Bidokhti. "[Neural Estimation of the Rate-Distortion Function With Applications to Operational Source Coding](https://arxiv.org/pdf/2204.01612.pdf)." arXiv preprint arXiv:2204.01612 (2022).

Eric Lei, Hamed Hassani, and Shirin Saeedi Bidokhti. "Neural Estimation of the Rate-Distortion Function For Massive Datasets," in 2022 IEEE International Symposium on Information Theory (ISIT), June 2022.

# To Run
Trained networks have been saved in the ``trained_lagr/`` folder. These can be used to plot RD curves in ``plotRDcurves_release.ipynb``. To train NERD from scratch on image datasets, it is easier to pretrain a GAN on the dataset first, and use the GAN to initialize the $Q_Y$ generator neural network. Trained GANs have been uploaded to the ``trained_gan/`` folder for MNIST, FMNIST, and SVHN datasets. To run NERD with these pretrained GANs, simply run ``bash scripts/NERD_{dataset}.sh``. If you wish to pretrain a GAN yourself, we have the ``wgan_gp.py`` file which trains a Wasserstein GAN with gradient penalty. 

Once you have trained NERD, the RCC methods can be run via ``bash scripts/RCC_{dataset}.sh``. 

# Dependencies
- torch
- numpy
- scipy
- pytorch-lightning==1.9.0
- pykeops
- huffman
- tqdm
- argparse
- matplotlib

# Citation

    @ARTICLE{lei2022neuralrd,
      author={Lei, Eric and Hassani, Hamed and Bidokhti, Shirin Saeedi},
      journal={IEEE Journal on Selected Areas in Information Theory}, 
      title={Neural Estimation of the Rate-Distortion Function With Applications to Operational Source Coding}, 
      year={2022},
      volume={3},
      number={4},
      pages={674-686},
      doi={10.1109/JSAIT.2023.3273467}
    }


