import argparse
from trainer import train
from evaluator import test


def main(args):
    args = vars(args)
    if args['test_only']:
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ACIL / DS-AL experiments unified management.'
    )

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

    # Basic options
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100_224',
        choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224'],
        help='Dataset to use',
    )
    parser.add_argument('--smart_defaults', default=False, action='store_true')
    parser.add_argument('--prefix', type=str, default='original')

    # Memory parameters
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--memory_per_class', type=int, default=0)
    parser.add_argument('--fixed_memory', default=False, action='store_true')
    parser.add_argument('--shuffle', default=True, action='store_true')

    # Class parameters
    parser.add_argument('--init_cls', type=int, default=20)
    parser.add_argument('--increment', type=int, default=20)

    # Model parameters
    parser.add_argument(
        '--model_name', type=str, default='dsal', choices=['acil', 'dsal']
    )
    parser.add_argument(
        '--vit_type',
        type=str,
        default='vit-b-p16-mocov3',
        choices=['vit-b-p16-lora-mocov3', 'vit-b-p16-lora', 'vit-b-p16-lora-mae']
    )

    parser.add_argument(
        '--lora_type', type=str, default='basic_lora', choices=['full', 'basic_lora']
    )
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--device', nargs='+', default=['0'])
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--num_used_layers', type=int, default=1)

    # Training parameters (mainly for logging consistency)
    parser.add_argument('--sce_a', type=float, default=0.5)
    parser.add_argument('--sce_b', type=float, default=0.5)
    parser.add_argument('--seed', nargs='+', type=int, default=[1993, 1996, 1997])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ca_epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--head_scale', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tune_classifier', default=True)
    parser.add_argument('--gamma_norm', type=float, default=0.1)
    parser.add_argument('--gamma_kd', type=float, default=1.0)
    parser.add_argument('--kd_type', type=str, default='feat')
    parser.add_argument('--only_lora', default=True, action='store_true')
    parser.add_argument('--user', type=str, default='null', choices=['null'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gamma_2', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=1.0)

    # Analytic baselines (ACIL / DS-AL)
    parser.add_argument('--random_feature_dim', type=int, default=8192)
    parser.add_argument('--ridge_lambda', type=float, default=1e-3)
    parser.add_argument('--rls_eps', type=float, default=1e-6)
    parser.add_argument(
        '--acil_activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu', 'tanh', 'mish'],
    )
    parser.add_argument(
        '--dsal_main_activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu', 'tanh', 'mish'],
    )
    parser.add_argument(
        '--dsal_comp_activation',
        type=str,
        default='tanh',
        choices=['relu', 'gelu', 'tanh', 'mish'],
    )
    parser.add_argument('--dsal_fusion_weight', type=float, default=1.0)

    # First-section adaptation options
    parser.add_argument(
        '--first_section_adaptation',
        dest='first_section_adaptation',
        type=bool,
        default=True
    )

    parser.add_argument(
        '--no_first_section_adaptation',
        dest='first_section_adaptation',
        action='store_false',
    )

    parser.set_defaults(first_section_adaptation=True)
    parser.add_argument('--fsa_steps', type=int, default=1000)
    parser.add_argument('--fsa_lr', type=float, default=1e-4)
    parser.add_argument('--fsa_weight_decay', type=float, default=0.0)
    parser.add_argument('--fsa_batch_size', type=int, default=32)

    args = parser.parse_args()
    main(args)