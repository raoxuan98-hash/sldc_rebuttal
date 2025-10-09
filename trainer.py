import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.multi_data_manager import MultiDataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    
    # Initialize storage for all results (dynamically populated per model)
    all_runs_results = {}
    
    for run_id, seed in enumerate(seed_list):
        args['seed'] = seed
        args['run_id'] = run_id
        args['device'] = device
        
        # Create log directory
        logfile_head = os.path.join(
            f"{args['model_name']}_0624_logs_{args['user']}",
            f"{args['dataset']}_{args['vit_type']}"
        )
        os.makedirs(logfile_head, exist_ok=True)
        
        
         # 判断单数据集或多数据集
        if isinstance(args['dataset'], list) and len(args['dataset']) > 1:
            # 多数据集模式，使用数据集名称
            ordered_datasets = [args['dataset'][i] for i in args['dataset_order']]
            dataset_names_str = "_".join(ordered_datasets)
            log_suffix = f"multi_dataset_{dataset_names_str}_"
        else:
            # 单数据集模式，使用 init_cls 和 increment
            log_suffix = f"init-{args['init_cls']}_inc-{args['increment']}_"
            
        logfile_name = os.path.join(
            logfile_head,
            f"{log_suffix}"
            f"optim-{args['optimizer']}_lr-{args['lrate']}_kd-{args['gamma_kd']}_"
            f"norm-{args['gamma_norm']}_type-{args['kd_type']}_tem-{args['alpha_t']}_"
            f"gamma_2-{args['gamma_2']}_"
            f"bz-{args['batch_size']}_epoch-{args['epochs']}_"
            f"seed-{args['seed']}"
        )

        
        if args['only_lora']:
            logfile_name += "_onlylora"
        
        os.makedirs(logfile_name, exist_ok=True)
        args['log_path'] = logfile_name
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        if seed == seed_list[0]:
            save_path = logfile_name
            
        results = train_single_run(args)
        
        # Append results for existing classifiers, filling missing entries with None
        for classifier in list(all_runs_results.keys()):
            if classifier in results:
                all_runs_results[classifier].append(results[classifier])
            else:
                all_runs_results[classifier].append(None)

        # Add new classifier keys introduced by this run
        for classifier, values in results.items():
            if classifier not in all_runs_results:
                all_runs_results[classifier] = [None] * run_id
                all_runs_results[classifier].append(values)
    
        # Calculate and save statistics
        stats = calculate_statistics(all_runs_results)
        save_results(all_runs_results, stats, save_path)
    
    return all_runs_results, stats

def train_single_run(args):
    seed = args['seed']
    set_random(seed)
    set_device(args)
    print_args(args)
    
    if isinstance(args['dataset'], list) and len(args['dataset']) > 1:
        data_manager = MultiDataManager(
            args['dataset'],
            args['dataset_order'],
            args['shuffle'],
            seed
        )
         # 动态设置 init_cls 和 increment
        increments = data_manager._increments  # 每个任务的类别数列表，例如 [100, 196, 200, 200]
        if increments:
          args['init_cls'] = increments[0]  # 第一个任务的类别数，例如 100（CIFAR-100）
          args['increment'] = increments[1:] if len(increments) > 1 else [0]  # 后续任务的增量，例如 [196, 200, 200]
        else:
          raise ValueError("No tasks defined in data_manager._increments")
    else:
        data_manager = DataManager(
            args['dataset'][0] if isinstance(args['dataset'], list) else args['dataset'],
            args['shuffle'],
            seed,
            args['init_cls'],
            args['increment']
        )
    
    model = factory.get_model(args['model_name'], args)
    logging.info(f'All params: {count_parameters(model._network)}')
    logging.info(f'Trainable params: {count_parameters(model._network, True)}')
    
    
    final_results = model.loop(data_manager)
    return final_results


def calculate_statistics(all_runs_results):
    """Calculate mean and std across all runs for each classifier and task"""
    stats = {}
    
    for classifier, runs in all_runs_results.items():
        valid_runs = [run for run in runs if run is not None]
        if not valid_runs:
            continue

        num_tasks = len(valid_runs[0])
        num_runs = len(runs)
        
        # Initialize arrays
        task_means = []
        task_stds = []
        
        for task_idx in range(num_tasks):
            task_values = []
            for run_idx in range(num_runs):
                if runs[run_idx] is None:
                    continue
                if runs[run_idx][task_idx] is not None:
                    task_values.append(runs[run_idx][task_idx])
            
            if task_values:
                task_mean = sum(task_values) / len(task_values)
                task_std = (sum((x - task_mean)**2 for x in task_values) / len(task_values))**0.5
                task_means.append(task_mean)
                task_stds.append(task_std)
            else:
                task_means.append(None)
                task_stds.append(None)
        
        stats[classifier] = {
            'mean': task_means,
            'std': task_stds
        }
    
    return stats

def save_results(all_runs_results, stats, output_dir):
    """Save all results and statistics to files"""
    import json
    
   
    with open(os.path.join(output_dir, 'all_runs_results.json'), 'w') as f:
        json.dump(all_runs_results, f, indent=2)
    
   
    with open(os.path.join(output_dir, 'results_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    
    summary = []
    for classifier, classifier_stats in stats.items():
        if not classifier_stats:
            continue
            
        summary.append(f"\nClassifier: {classifier}")
        for task_idx in range(len(classifier_stats['mean'])):
            mean = classifier_stats['mean'][task_idx]
            std = classifier_stats['std'][task_idx]
            
            if mean is not None and std is not None:
                summary.append(f"Task {task_idx+1}: Mean = {mean:.2f}%, Std = {std:.2f}")
    
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write("\n".join(summary))
    
    logging.info("\nResults Summary:")
    logging.info("\n".join(summary))


def set_device(args):
    """Properly set the device based on args"""
    device_type = args['device']
    if isinstance(device_type, (list, tuple)):
        gpus = []
        for device in device_type:
            if device == -1:
                gpus.append(torch.device('cpu'))
            else:
                gpus.append(torch.device(f'cuda:{device}'))
        args['device'] = gpus
    else:
        if device_type == -1:
            args['device'] = torch.device('cpu')
        else:
            args['device'] = torch.device(f'cuda:{device_type}')
def set_random(seed):
    """Set all random seeds using the provided seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
