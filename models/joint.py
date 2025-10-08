import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
import copy
from utils.inc_net import FinetuneIncrementalNet
from collections import OrderedDict
import time
from utils.toolkit import tensor2numpy, accuracy

num_workers = 3
eval_freq = 1000
num_used_layers = 1
use_dirft_compenstation = True

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss


class JointLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args, pretrained=True, num_used_layers=num_used_layers)
        self.model_name_= args['model_name']
        self.args = args

        if 'log_path' in args.keys():
            self.log_path = args['log_path'] 

        self.model_prefix = args['prefix']
        self.sce_a, self.sce_b = args['sce_a'], args['sce_b']

        for n, p in self._network.convnet.named_parameters():
            if args['only_lora']:
                if  'lora' in n:
                    p.requires_grad=True
                else:
                    p.requires_grad=False
            else:
                if 'norm' in n or 'bias' in n or 'lora' in n or "cls_token" in n:
                    p.requires_grad=True
                else:
                    p.requires_grad=False

        if args['test_only']:
            pass
        
        else:
            for b_idx in range(self._network.convnet.lora_lp):
                self._network.convnet.blocks[b_idx].mlp.init_lora()
                self._network.convnet.blocks[b_idx].attn.init_lora()
        
        self._network.cuda()

        if 'weight_decay' in args.keys():
            global weight_decay
            weight_decay = args['weight_decay']

        if 'lrate' in args.keys():
            global lrate
            lrate = args['lrate']

        if "epochs" in args.keys():
            global epochs
            epochs = args['epochs']
        
        if "ca_epochs" in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs']

        if 'optimizer' in args.keys():
            global optimizer_type
            optimizer_type = args['optimizer']

        if 'head_scale' in args.keys():
            global head_scale
            head_scale = args['head_scale']

        if 'batch_size' in args.keys():
            global batch_size
            batch_size = args['batch_size']
        
        self.run_id = args['run_id']
        self.seed = args['seed']

        self.task_sizes = []
        self.training_times = []

    def save_checkpoint(self, filename):
        param_dict = {n: p.detach().cpu() for n, p in self._network.named_parameters() if p.requires_grad}
        to_save = {'task': self._cur_task, "model_state_dict": param_dict}
        torch.save(to_save, f'{filename}_after_task_{self._cur_task}.pth')
        
    
    def train(self, data_manager):
        start_time = time.time()  
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self.topk = self._total_classes if self._total_classes < 5 else 5

        """数据集处理"""
        train_dset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train', mode='train', appendent=[], with_raw=False)
        
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        """系统学习"""
        self._network.update_fc(task_size, freeze_old=False)
        self._network.fc.to(self._device)
        logging.info('System training on {}-{} of {}'.format(self._known_classes, self._total_classes, dset_name))
        self._system_training(self.train_loader, self.test_loader)
        total_time = time.time() - start_time

        self.training_times.append({
            "task": self._cur_task,
            "total": total_time
        })

        logging.info(f"Task {self._cur_task} Training Time Breakdown:")
        logging.info(f"  - Total Training Time: {total_time:.2f}s")


    def get_optimizer(self, network_params, lr, weight_decay):
        if optimizer_type == 'sgd':
            return optim.SGD(network_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(network_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(network_params, lr=lr, weight_decay=weight_decay)

    def _system_training(self, train_loader, test_loader):
        base_lora_params_slow = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora' in n]
        base_others_params_slow = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora' not in n]
        base_fc_params_slow = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        
        base_lora_params_slow = {'params': base_lora_params_slow, 'lr': lrate, 'weight_decay': weight_decay}
        base_others_params_slow = {'params': base_others_params_slow, 'lr': lrate, 'weight_decay': weight_decay}
        base_fc_params_slow = {'params': base_fc_params_slow, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
        
        network_params_slow = [base_lora_params_slow, base_others_params_slow, base_fc_params_slow]

        optimizer_slow = self.get_optimizer(network_params_slow, lrate, weight_decay)
        scheduler_slow = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_slow, T_max=epochs, eta_min=lrate/3)
        
        self._system_run(train_loader, test_loader, optimizer_slow, scheduler_slow)

    def calculate_kd_loss(self, previous_features, current_features):
        kd_loss = self.kd_loss(previous_features, current_features)
        return kd_loss
    
    def calculate_norm_loss(self, previous_features, current_features):
        norms_t = torch.norm(previous_features, p=2, dim=1)
        norms_s = torch.norm(current_features, p=2, dim=1)
        norm_loss = F.mse_loss(norms_t, norms_s)
        return norm_loss

    def _system_run(self, train_loader, test_loader, optimizer_slow, scheduler_slow):
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
           
            self._network.train()
            losses, correct_slow, total = 0.0, 0, 0
            # Initialize all loss trackers
            rce_loss_slows = 0.0
            steps_per_epoch = len(train_loader)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                slow_features = self._network.convnet(inputs)['features']
                slow_logits = self._network.fc(slow_features)
                cur_targets = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -100)
                cur_slow_logits = slow_logits[:, self._known_classes:]
                rce_loss_slow = symmetric_cross_entropy_loss(cur_slow_logits, cur_targets, self.sce_a, self.sce_b)                
                rce_loss_slows += rce_loss_slow.item()
                loss = rce_loss_slow
                optimizer_slow.zero_grad()
                loss.backward()           
                optimizer_slow.step()
                losses += loss.item()
                correct_slow += torch.sum(torch.argmax(slow_logits, dim=1) == targets)
                total += targets.size(0)

                if i % 100 == 0:
                    avg_loss = losses / len(train_loader)
                    train_acc_slow = correct_slow / total
                    info = (f'Task {self._cur_task}, Slow System Training, Epoch {epoch}/{run_epochs}, Step {i}/{steps_per_epoch} => '
                            f'Total Loss: {avg_loss:.3f}, '
                            f'Train Acc: {train_acc_slow:.3f}')
                    logging.info(info)
                    print(info)

            scheduler_slow.step()

    def loop(self, data_manager):
        self.train(data_manager)
        results = self.evaluate()
        info = f"Evaluation accuracy: {results:.2f}"
        logging.info(info)
        return results

    def evaluate(self):
        self._network.eval()
        total = 0 
        correct_counts = 0 
        for _, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            targets_np = targets.cpu().numpy()
            total += len(targets_np)
            with torch.no_grad():
                features = self._network.convnet(inputs)['features']
                preds = self._network.fc(features)
                preds = preds.argmax(dim=1).cpu().numpy()
                correct_counts += (preds == targets_np).sum()
        return np.around(100.0 * correct_counts / total, decimals=2)
