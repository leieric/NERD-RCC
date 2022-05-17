import NERD
import models
import dataloaders
import os
from argparse import ArgumentParser, Namespace

def RD_sweep(args, generator, datamodule):
    for D in args.Ds[0]:
        NERD.train_save(args, D, generator, datamodule)
    



if __name__ == '__main__':
    
    
    parser = ArgumentParser()
#     parser.add_argument("--gpus", type=int, default=[0], help="gpu list")
    parser.add_argument('-g','--gpus', type=int, nargs='+', action='append', help='gpu_list')
    parser.add_argument('--Ds', type=float, nargs='+', action='append', help='Distortion points')
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="G learning rate")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--data_name", type=str, default="MNIST", help="dataset name")
    

    args = parser.parse_args()
    
    if not os.path.exists(f'trained/figures_{args.data_name}'):
        os.mkdir(f'trained/figures_{args.data_name}')
    if not os.path.exists(f'trained/trained_{args.data_name}'):
        os.mkdir(f'trained/trained_{args.data_name}')
        
    if args.data_name == "MNIST":
        dm = dataloaders.MNISTDataModule(args.batch_size)
        generator = models.Generator(img_size=(32,32,1), latent_dim=args.latent_dim, dim=64)
    elif args.data_name == "FMNIST":
        dm = dataloaders.FMNISTDataModule(args.batch_size)
        generator = models.Generator(img_size=(32,32,1), latent_dim=args.latent_dim, dim=64)
    elif args.data_name == "Gaussian":
        m = 20
        r = 0.25
        dm = dataloaders.GaussianDataModule(args.batch_size, m, r)
        generator = models.Decoder_FC(m, args.latent_dim)
    

    RD_sweep(args, generator, dm)