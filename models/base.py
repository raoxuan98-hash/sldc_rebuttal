import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from collections import OrderedDict

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args['memory_size']
        self._memory_per_class = args['memory_per_class']
        self._fixed_memory = args['fixed_memory']
        self._device = args['device'][0]
        self._multiple_gpus = args['device']
        
        self._init_cls = args['init_cls']
        
      #  self._increments = args['increment']  # 后续任务的增量列表
        
        if isinstance(args['increment'], list):  # 混合数据集
            self._increments = args['increment']  # 后续任务的类别数列表
            self._incrment_cls = None  # 对于混合数据集，不使用固定增量
        else:  # 单个数据集
            self._increments = None
            self._incrment_cls = args['increment']  # 固定增量
            
        self._incrment_cls = args['increment']


        # Metric tracking for CIL
        self.task_count = 1
        self.accuracy_matrix = []  # A_{t', t}: list of lists, each column is accuracies on all tasks after task t
        self.max_accuracy = []     # max_{τ \ge t'} A_{t', \u03C4} for each task

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return (self._memory_size // self._total_classes)

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def save_checkpoint(self, filename, head_only=False, learnable_only=False):
        if hasattr(self._network, 'module'):
            to_save = self._network.module
        else:
            to_save = self._network

        if head_only:
            to_save_dict = to_save.fc.state_dict()
        else:
            to_save_dict = to_save.state_dict()
            
        if learnable_only:
            new_dict = OrderedDict()
            filtered_keys = [n for n, p in to_save.named_parameters() if p.requires_grad]
            for k in filtered_keys:
                new_dict[k] = to_save_dict[k]
            to_save_dict = new_dict

        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': to_save_dict,
        }
        
        torch.save(save_dict, f'{filename}_{self._cur_task}.pth')

    def after_task(self):
        # increment task counter
        self.task_count += 1

    def _evaluate(self, y_pred, y_true):
        """
        Evaluate predictions and track CIL metrics:
          - A_{t', t}: accuracy on task t' after current task t
          - Avg-Acc_t:    average over all encountered tasks after task t
          - New-Acc_t:    average of accuracies when each task was first learned
          - Forget_t:     average forgetting over all previous tasks
        """
        
        if self._increments is not None:  # 混合数据集
           if self._cur_task == -1:  # 未开始训练，使用 init_cls
               num_classes_per_task = self._init_cls
           else:
               num_classes_per_task = self._increments[self._cur_task - 1] if self._cur_task > 0 else self._init_cls
        else:  # 单个数据集
          num_classes_per_task = self._incrment_cls
        
        # 1) compute per-task accuracy
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, num_classes_per_task)
        per_task_acc = grouped['class_acc']  # list of length = #tasks seen so far

        # 2) append this new column A_{:, t}
        self.accuracy_matrix.append(per_task_acc)

        # 3) update max‐ever accuracies, *extending* if needed*
        if not self.max_accuracy:
            self.max_accuracy = per_task_acc.copy()
        else:
            new_max = []
            for idx, acc in enumerate(per_task_acc):
                if idx < len(self.max_accuracy):
                    new_max.append(max(self.max_accuracy[idx], acc))
                else:
                    # brand-new task slot: take the new acc
                    new_max.append(acc)
            self.max_accuracy = new_max


        # 4) now compute t = number of tasks learned so far
        t = len(self.accuracy_matrix)    # guaranteed ≥ 1

        # 5) Avg-Acc_t = (1/t) * sum over tasks 1…t of A_{t',t}
        avg_acc = sum(per_task_acc) / t

        # 6) New-Acc_t = (1/t) * sum of diagonal A_{t',t'}
        new_acc = sum(self.accuracy_matrix[i][i] for i in range(t)) / t

        # 7) Forget_t = (1/(t-1)) * sum over tasks 1…t-1 of (max_ever – current)
        if t > 1:
            forget = (
                sum(
                    self.max_accuracy[i]
                    - self.accuracy_matrix[-1][i]
                    for i in range(t - 1)
                )
                / (t - 1)
            )
        else:
            forget = 0.0

        # 8) package up standard top-1 / top-k as before
        ret = {
            'grouped': grouped,
            'top1': grouped['total'],
            f'top{self.topk}': np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
                decimals=2
            ),
            # and your CIL metrics:
            'A_column':      per_task_acc,
            'Avg-Acc':       avg_acc,
            'New-Acc':       new_acc,
            'Forget':        forget,
        }
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _inner_eval(self, model, loader):
        model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]       

        cnn_accy = self._evaluate(y_pred, y_true) 
        return cnn_accy

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        return np.concatenate(y_pred), np.concatenate(y_true)
    def _extract_vectors_aug(self, loader, repeat=2):
        self._network.eval()
        vectors, targets = [], []
        for _ in range(repeat):
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                with torch.no_grad():
                    if isinstance(self._network, nn.DataParallel):
                        _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
                    else:
                        _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
            
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

            # self._class_covs = []

        if check_diff:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                # vectors, _ = self._extract_vectors_aug(idx_loader)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                # class_cov = np.cov(vectors.T)
                class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                    logging.info(log_info)
                    np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                    # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))

        if oracle:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                # class_cov = np.cov(vectors.T)
                class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-5
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov            

        for class_idx in range(self._known_classes, self._total_classes):
            # data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
            #                                                       mode='train', ret_data=True)
            # idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            # vectors_aug, _ = self._extract_vectors_aug(idx_loader)

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            if check_diff:
                log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                logging.info(log_info)
                np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self._cur_task, class_idx), self._class_means[class_idx, :])
                # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov
            # self._class_covs.append(class_cov)