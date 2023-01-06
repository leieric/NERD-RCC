import NERDlagr as NERD
import models
import dataloaders
import os
from argparse import ArgumentParser, Namespace

def RD_sweep(args, generator, datamodule):
    for lmbda in args.lmbdas[0]:
        NERD.train_save(args, lmbda, generator, datamodule)
    



if __name__ == '__main__':
    
    
    parser = ArgumentParser()
#     parser.add_argument("--gpus", type=int, default=[0], help="gpu list")
    parser.add_argument('-g','--gpus', type=int,nargs='+', action='append', help='gpu_list')
    parser.add_argument('--lmbdas', type=float, nargs='+', action='append', help='Distortion points')
    parser.add_argument("--batch_size", type=int, default=2000, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="G learning rate")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="dimensionality of the latent space")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--data_name", type=str, default="MNIST", help="dataset name")
    parser.add_argument("--init_gan", type=int, default=0, help="init with trained GAN")

    args = parser.parse_args()
    
    if not os.path.exists(f'trained_lagr/figures_{args.data_name}'):
        os.mkdir(f'trained_lagr/figures_{args.data_name}')
    if not os.path.exists(f'trained_lagr/trained_{args.data_name}'):
        os.mkdir(f'trained_lagr/trained_{args.data_name}')
        
    if args.data_name == "MNIST":
        dm = dataloaders.MNISTDataModule(args.batch_size)
        args.dnn_size=32
        # generator=None
        generator = models.Generator(img_size=(32,32,1), latent_dim=args.latent_dim, dim=args.dnn_size)
    elif args.data_name == "FMNIST":
        dm = dataloaders.FMNISTDataModule(args.batch_size)
        args.dnn_size=32
        generator = models.Generator(img_size=(32,32,1), latent_dim=args.latent_dim, dim=args.dnn_size)
    elif args.data_name == "SVHN":
        dm = dataloaders.SVHNDataModule(args.batch_size)
        args.dnn_size=32
        generator = models.Generator(img_size=(32,32,3), latent_dim=args.latent_dim, dim=args.dnn_size)
    elif args.data_name == "Gaussian":
        m = 20
        r = 0.25
        dm = dataloaders.GaussianDataModule(args.batch_size, m, r)
        generator = models.Decoder_FC(m, args.latent_dim)
    elif args.data_name == "Gaussian2":
        m = 40
        r = 0.5
        dm = dataloaders.GaussianDataModule(args.batch_size, m, r)
        generator = models.Decoder_FC(m, args.latent_dim)
    elif args.data_name == "Sawbridge":
        dm = dataloaders.Sawbridge(args.batch_size, n=10000000, n_sample=1024)
        generator = models.Decoder_FC(1024, 10)
    

    RD_sweep(args, generator, dm)