import argparse
from trainer import train
from evaluator import test

# In[]
def main(args):
    args = vars(args)  # Converting argparse Namespace to a dict.
    if args['test_only']:
        test(args)
    else:
        train(args)

def set_smart_defaults(args):
    if args.smart_defaults:
        if args.dataset == 'cars196_224':
            args.init_cls = 20
            args.increment = 20
            args.epochs = 15

        elif args.dataset == 'imagenet-r':
            args.init_cls = 20
            args.increment = 20
            args.epochs = 10
        
        elif args.dataset == 'cifar100_224':
            args.init_cls = 10
            args.increment = 10     
            args.epochs = 5

        elif args.dataset == 'cub200_224':
            args.init_cls = 20
            args.increment = 20
            args.epochs = 15
    return args
# In[]
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLDC experiments unified management.')  
    
    # Basic options
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='imagenet-r', choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224'], help='Dataset to use')
    parser.add_argument('--smart_defaults', default=False, action='store_true')
    parser.add_argument('--prefix', type=str, default="original")
    
    # Memory parameters  
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--memory_per_class', type=int, default=0)
    parser.add_argument('--fixed_memory', default=False, action='store_true')
    parser.add_argument('--shuffle', default=True, action='store_true')
    
    # Class parameters
    parser.add_argument('--init_cls', type=int, default=20)
    parser.add_argument('--increment', type=int, default=20)
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='acil', choices=['sldc', 'acil', 'dsal'])
    parser.add_argument('--vit_type', type=str, default='vit-b-p16-mocov3', choices=['vit-b-p16-lora-mocov3', 'vit-b-p16-lora', 'vit-b-p16-lora-mae'])
    parser.add_argument('--lora_type', type=str, default='basic_lora', choices=['full', 'basic_lora'])
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--device', nargs='+', default=['0'])
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--num_used_layers', type=int, default=1)
    
    # Training parameters
    parser.add_argument('--sce_a', type=float, default=0.5)
    parser.add_argument('--sce_b', type=float, default=0.5)
    parser.add_argument('--seed', nargs='+', type=int, default=[1990, 1996, 1997])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ca_epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    # parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--head_scale', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tune_classifier', default=True)
    parser.add_argument('--gamma_norm', default=0.1, type=float)
    parser.add_argument('--gamma_kd', type=float, default=1.0)
    parser.add_argument('--kd_type', type=str, default='feat')
    parser.add_argument('--only_lora', default=True, action='store_true')
    parser.add_argument('--user', type=str, default='null', choices=['null'])
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--use_linear_compensation', type=bool, default=False)
    parser.add_argument('--use_weak_nonlinear_compensation', type=bool, default=True)
    parser.add_argument('--use_mlp_compensation', type=bool, default=False)
    parser.add_argument('--gamma_1', type=float, default=1e-4)
    # parser.add_argument('--gamma_2', type=float, default=0.5)
    parser.add_argument('--gamma_2', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=1.0)

    parser.add_argument('--use_auxiliary_data_enhancement', type=bool, default=True)
    parser.add_argument('--auxiliary_data_path', type=str, default="/data1/open_datasets/ImageNet-2012/train/")
    parser.add_argument('--auxiliary_data_size', type=int, default=1024)

    # Analytic baselines (ACIL / DS-AL)
    parser.add_argument('--random_feature_dim', type=int, default=8192)
    parser.add_argument('--ridge_lambda', type=float, default=1e-3)
    parser.add_argument('--rls_eps', type=float, default=1e-6)
    parser.add_argument('--acil_activation', type=str, default='relu')
    parser.add_argument('--dsal_main_activation', type=str, default='relu')
    parser.add_argument('--dsal_comp_activation', type=str, default='tanh')
    parser.add_argument('--dsal_fusion_weight', type=float, default=1.0)

    args = parser.parse_args()
    args = set_smart_defaults(args)  # Apply smart defaults
    main(args)
