import NERD
import os
from argparse import ArgumentParser, Namespace

def RD_sweep(args):
    for D in args.Ds[0]:
        NERD.train_save(args, D)
    



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
    

    RD_sweep(args)