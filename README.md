# NERD-RCC
Neural estimation of the rate-distortion function and its applications to reverse channel coding and compression. Provides a neural estimator (NERD) for estimating the rate-distortion function R(D) from i.i.d. samples. Uses NERD to implement single-shot lossy compression with guarantees on achievable rate-distortion.

For details, see:

Eric Lei, Hamed Hassani, and Shirin Saeedi Bidokhti. "[Neural Estimation of the Rate-Distortion Function With Applications to Operational Source Coding](https://arxiv.org/pdf/2204.01612.pdf)." arXiv preprint arXiv:2204.01612 (2022).

Eric Lei, Hamed Hassani, and Shirin Saeedi Bidokhti. "Neural Estimation of the Rate-Distortion Function For Massive Datasets," in 2022 IEEE International Symposium on Information Theory (ISIT), June 2022.

# To Run
First, make sure you have folders named `trained/` and `data/` in the root directory. To estimate R(D) at a series of distortion values, run `python NERD_curve.py --gpus 0 --Ds 10 20 30 40 50 --data_name "MNIST"`. This will evaluate R(D) at D=10, 20, 30, 40, 50, using GPU 0 on MNIST. Then use `plotRDcurves.ipnyb` to plot the estimated curve.

# Citation

    @article{lei2022neuralrd,
        title = {Neural Estimation of the Rate-Distortion Function With Applications to Operational Source Coding},
        author = {Lei, Eric and Hassani, Hamed and Bidokhti, Shirin Saeedi},
        journal = {arXiv preprint arXiv:2204.01612},
        year = {2022},
    }


