import sys
import logging
import copy
import os
from typing import Dict, List, Any

from trainer import (
    train_single_run as _base_train_single_run,
    calculate_statistics,
    save_results,
)


def train(args: Dict[str, Any]):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    all_runs_results: Dict[str, List[Any]] = {}

    for run_id, seed in enumerate(seed_list):
        args['seed'] = seed
        args['run_id'] = run_id
        args['device'] = device

        log_root = _build_log_root(args)
        os.makedirs(log_root, exist_ok=True)

        log_dir = _build_log_dir(args, log_root)
        os.makedirs(log_dir, exist_ok=True)
        args['log_path'] = log_dir

        _reset_logging_handlers()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=os.path.join(log_dir, 'record.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )

        if seed == seed_list[0]:
            save_path = log_dir

        results = train_single_run(args)

        for classifier in list(all_runs_results.keys()):
            if classifier in results:
                all_runs_results[classifier].append(results[classifier])
            else:
                all_runs_results[classifier].append(None)

        for classifier, values in results.items():
            if classifier not in all_runs_results:
                all_runs_results[classifier] = [None] * run_id
                all_runs_results[classifier].append(values)

        stats = calculate_statistics(all_runs_results)
        save_results(all_runs_results, stats, save_path)

    return all_runs_results, stats


def train_single_run(args: Dict[str, Any]):
    base_results = _base_train_single_run(args)
    return _append_avg_acc(base_results)


def _append_avg_acc(results: Dict[str, List[Any]]):
    augmented: Dict[str, List[Any]] = {}

    for key, values in results.items():
        # Preserve original metrics
        augmented[key] = list(values)

        if isinstance(values, list) and values:
            augmented[f"{key}_avg_acc"] = _compute_running_average(values)
        else:
            augmented[f"{key}_avg_acc"] = []

    return augmented


def _compute_running_average(values: List[Any]) -> List[Any]:
    running_sum = 0.0
    count = 0
    averages: List[Any] = []

    for value in values:
        if value is None:
            averages.append(None)
            continue
        running_sum += float(value)
        count += 1
        averages.append(round(running_sum / count, 2))

    return averages


def _reset_logging_handlers():
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()


def _build_log_root(args: Dict[str, Any]) -> str:
    dataset = args['dataset']
    if isinstance(dataset, list):
        dataset_name = 'multi-' + '_'.join(dataset)
    else:
        dataset_name = dataset

    vit_type = args.get('vit_type', 'na')
    return os.path.join(
        f"{args['model_name']}_0624_logs_{args['user']}",
        f"{dataset_name}_{vit_type}"
    )


def _build_log_dir(args: Dict[str, Any], log_root: str) -> str:
    parts = [
        f"init-{args['init_cls']}",
        f"inc-{_format_increment(args['increment'])}",
        f"opt-{args['optimizer']}",
        f"lr-{_format_float(args['lrate'])}",
        f"wd-{_format_float(args.get('weight_decay', 0.0))}",
        f"lora-{args.get('lora_type', 'na')}",
        f"rank-{args.get('lora_rank', 'na')}",
        f"layers-{args.get('num_used_layers', 'na')}",
        f"bs-{args['batch_size']}",
        f"epoch-{args['epochs']}",
        f"seed-{args['seed']}",
    ]

    if args['model_name'] == 'acil':
        parts.extend([
            f"randfeat-{args.get('random_feature_dim', 'na')}",
            f"ridge-{_format_float(args.get('ridge_lambda', 0.0))}",
            f"rls-{_format_float(args.get('rls_eps', 0.0))}",
            f"act-{args.get('acil_activation', 'na')}",
        ])
    elif args['model_name'] == 'dsal':
        parts.extend([
            f"mainact-{args.get('dsal_main_activation', 'na')}",
            f"compact-{args.get('dsal_comp_activation', 'na')}",
            f"fusion-{_format_float(args.get('dsal_fusion_weight', 0.0))}",
        ])

    if args.get('first_section_adaptation', False):
        parts.append(
            f"fsa-{args.get('fsa_steps', 'na')}-lr-{_format_float(args.get('fsa_lr', 0.0))}"
        )
    else:
        parts.append('nofsa')

    if args.get('only_lora'):
        parts.append('onlylora')

    return os.path.join(log_root, '_'.join(str(part) for part in parts))


def _format_increment(increment: Any) -> str:
    if isinstance(increment, (list, tuple)):
        return '-'.join(str(item) for item in increment)
    return str(increment)


def _format_float(value: Any) -> str:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if float_value == int(float_value):
        return str(int(float_value))
    return f"{float_value:g}"
