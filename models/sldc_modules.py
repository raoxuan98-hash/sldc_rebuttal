import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import copy
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

class WeightedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_scale=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_scale = use_scale
        
        if use_scale:
            self.scale = nn.Parameter(torch.ones(out_features))
        else:
            self.scale = torch.ones(1)
    def forward(self, x):
        x = self.linear(x)
        return x * self.scale

def cholesky_manual_parallel(matrix):
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))
        if j < n - 1:
            s_off = torch.mm(L[j+1:, :j], L[j, :j].unsqueeze(1)).squeeze(1)
            L[j+1:, j] = (matrix[j+1:, j] - s_off) / L[j, j]
    return L

def sample_torch_cholesky(n_samples, mean, L, given_Z=None):
    if given_Z is not None:
        random_indices = torch.randperm(given_Z.shape[0] )
        Z = given_Z[random_indices][0:n_samples].to(mean.device)
    else:
        Z = torch.randn(n_samples, mean.size(0), device=mean.device)
    X = Z @ L.T + mean.unsqueeze(0)
    return X

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

class GaussianStatistics:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.L = cholesky_manual_parallel(cov + 1e-3 * torch.eye(cov.size(0), device=cov.device))
        
        self.mean = self.mean.cpu()
        self.cov = self.cov.cpu()
        self.L = self.L.cpu()
        
    def kl_divergence(self, other, eps=1e-6):
        """计算KL散度"""
        d = self.mean.size(0)
        cov2_inv = torch.linalg.inv(other.cov + eps * torch.eye(d, device=other.cov.device))
    
        diff = self.mean - other.mean
        kl = 0.5 * (
            torch.logdet(other.cov + eps * torch.eye(d, device=other.cov.device)) -
            torch.logdet(self.cov + eps * torch.eye(d, device=self.cov.device)) +
            torch.trace(cov2_inv @ self.cov) + diff @ cov2_inv @ diff - d)
        return kl
    
class NonlinearCompensator(nn.Module):
    def __init__(self, dim):
        super(NonlinearCompensator, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.fc1.weight)

        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=True))
        
        self.alphas =  nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.weight = 0.0

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=0)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y = weights[0]*x + weights[1] * y1 + weights[2] * y2
        return (1.0 - self.weight) * y + self.weight * x
    
    def reg_loss(self):
        weights = F.softmax(self.alphas, dim=0)
        return (weights[0] + weights[1] - 1.0) ** 2
    
class SLDC_Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_linear_compensation = args['use_linear_compensation']
        self.use_weak_nonlinear_compensation = args['use_weak_nonlinear_compensation']
        self.use_mlp_compensation = args['use_mlp_compensation']
        # 原始统计信息（不使用辅助数据）
        self.original_stats = {}
        self.linear_stats = {}
        self.weak_nonlinear_stats = {}
        self.mlp_stats = {}
        
        # 使用辅助数据的统计信息
        self.linear_stats_with_aux = {}
        self.weak_nonlinear_stats_with_aux = {}
        self.mlp_stats_with_aux = {}

        self.alpha_t = args['alpha_t']
        self.gamma_1 = args['gamma_1']
        self.gamma_2 = args['gamma_2']
        self.auxiliary_data_size = args['auxiliary_data_size'] if args['use_auxiliary_data_enhancement'] else 0
        self.args = args
        self.use_auxiliary_data = args['use_auxiliary_data_enhancement']


    def update_stats(self, task_id, model_before, model_after, data_loader):
        # 提取原始特征
        feats_before, feats_after, targets = self.extract_features_before_after(model_before, model_after, data_loader)
        
        if task_id == 0:
            try:
                self.cached_Z = torch.load('cached_Gaussian_samples.pt')
            except:
                self.cached_Z = torch.randn(50000, feats_before.size(1))
                torch.save(self.cached_Z, 'cached_Gaussian_samples.pt')

        # 更新不使用辅助数据的统计信息
        original_stats = self.compute_class_statistics(feats_after, targets)
        assert set(original_stats.keys()).isdisjoint(set(self.original_stats.keys())), \
            "original_stats keys should not overlap with self.original_stats keys"
        self.original_stats.update(original_stats)
        
        # 如果使用辅助数据，则同时更新使用辅助数据的统计信息
        if self.use_auxiliary_data:
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after = self.extract_features_before_after_for_auxiliary_data(
                model_before, model_after, aux_loader)
    
            feats_before_with_aux = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_after_with_aux = torch.cat([feats_after, feats_aux_after], dim=0)
            
        if task_id > 0:
            if self.use_linear_compensation:
                self.linear_stats = self.update_statistics_with_linear_transform(
                    self.linear_stats, feats_before, feats_after)
                if self.use_auxiliary_data:
                    self.linear_stats_with_aux = self.update_statistics_with_linear_transform(
                        self.linear_stats_with_aux, feats_before_with_aux, feats_after_with_aux)
                    
            if self.use_weak_nonlinear_compensation:
                self.weak_nonlinear_stats = self.update_statistics_with_weak_nonlinear_transform(
                    self.weak_nonlinear_stats, feats_before, feats_after)
                if self.use_auxiliary_data:
                    self.weak_nonlinear_stats_with_aux = self.update_statistics_with_weak_nonlinear_transform(
                        self.weak_nonlinear_stats_with_aux, feats_before_with_aux, feats_after_with_aux)
                    
            if self.use_mlp_compensation:
                self.mlp_stats = self.update_statistics_with_mlp_transform(
                    self.mlp_stats, feats_before, feats_after)
                if self.use_auxiliary_data:
                    self.mlp_stats_with_aux = self.update_statistics_with_mlp_transform(
                        self.mlp_stats_with_aux, feats_before_with_aux, feats_after_with_aux)
                    
        # 更新当前任务的统计信息
        self.linear_stats.update(original_stats)
        self.weak_nonlinear_stats.update(original_stats)
        self.mlp_stats.update(original_stats)
        
        if self.use_auxiliary_data:
            self.linear_stats_with_aux.update(original_stats)
            self.weak_nonlinear_stats_with_aux.update(original_stats)
            self.mlp_stats_with_aux.update(original_stats)
        

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        """对比模型训练前后的特征变化"""
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        
        feats_before, feats_after, targets, indices = [], [], [], []
        
        for batch_indices, inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            feats_before.append(model_before.forward_features(inputs).cpu())
            feats_after.append(model_after.forward_features(inputs).cpu())
            targets.append(batch_targets)
            indices.append(batch_indices)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        indices = torch.cat(indices)
        return feats_before[indices], feats_after[indices], targets[indices]

    @torch.no_grad()
    def extract_features_before_after_for_auxiliary_data(self, model_before, model_after, data_loader):
        """对比模型训练前后的特征变化"""
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        
        feats_before, feats_after= [], []
        for inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            feats_before.append(model_before.forward_features(inputs).cpu())
            feats_after.append(model_after.forward_features(inputs).cpu())

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        return feats_before, feats_after
    
    def compute_class_statistics(self, features, labels):
        unique_labels = torch.unique(labels)
        stats_dict = {}
        for lbl in unique_labels:
            mask = (labels == lbl)
            class_features = features[mask]
            mean = class_features.mean(dim=0)
            centered = class_features - mean
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            stats_dict[int(lbl.item())] = GaussianStatistics(mean, cov)
        return stats_dict
    
    def update_statistics_with_linear_transform(self, stats, features_before, features_after):
        print("基于当前任务的前后特征构建线性补偿器(alpha_1-SLDC)")
        features_before = features_before.to(self.device)
        features_after = features_after.to(self.device)
        X = F.normalize(features_before, dim=1)
        Y = F.normalize(features_after, dim=1)
        XTX = X.T @ X + self.gamma_1 * torch.eye(X.size(1), device=self.device)
        XTY = X.T @ Y
        W_global = torch.linalg.solve(XTX, XTY)

        dim = features_before.size(1)
        sample_num = features_before.size(0)

        weight = math.exp(- sample_num / (self.alpha_t*dim))
        print(weight)
        W_global = (1 - weight)  * W_global + weight * torch.eye(dim, device=self.device)
        feats_new_after_pred = features_before @ W_global
        feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()

        s = torch.linalg.svdvals(W_global)
        max_singular = s[0].item()
        min_singular = s[-1].item()
        print(f"线性变换矩阵对角线元素平均值：{W_global.diag().mean().item():.4f}, 加权权重：{weight:.4f}, 样本数量：{sample_num}")
        print(f"线性修正前特征差异:{feat_diffs:.4f}; 修正后差异:{feat_diffs_pred:.4f}; 变换矩阵最大奇异值:{max_singular:.2f}; 最小奇异值:{min_singular:.2f}")

        updated_stats = {}
        for class_id, gauss in stats.items():
            old_mean = gauss.mean.to(self.device)
            old_cov = gauss.cov.to(self.device)
            new_mean = old_mean @ W_global
            new_cov = W_global.T @ old_cov @ W_global + 1e-2 * torch.eye(old_cov.size(0), device=old_cov.device)
            updated_stats[class_id] = GaussianStatistics(new_mean, new_cov)
        return updated_stats

    def update_statistics_with_weak_nonlinear_transform(self, stats, features_before, features_after):
        features_before = features_before.to(self.device)
        features_after = features_after.to(self.device)
        print("基于当前任务的前后特征构建弱非线性补偿器")
        mapping, alpha = self.optimize_nonlinear_transform(features_before, features_after)
        with torch.no_grad():
            feats_new_after_pred = mapping(F.normalize(features_before, dim=1)) * features_before.norm(dim=1, keepdim=True)
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()
        print(f"弱非线性修正后特征差异: {feat_diffs_pred:.4f}, Alpha: {alpha}")      
        with torch.no_grad():
            new_stats = {}
            for class_id, gauss in stats.items():
                class_mean = gauss.mean.to(self.device)
                class_L = gauss.L.to(self.device)
                d = class_mean.size(0)
                num = d * 10
                class_samples = sample_torch_cholesky(num, class_mean, class_L, self.cached_Z)
                class_samples = mapping(F.normalize(class_samples, dim=1)) * class_samples.norm(dim=1, keepdim=True)
                new_mean = class_samples.mean(dim=0)
                new_mean = new_mean
                new_cov = torch.cov(class_samples.T) + 1e-3 * torch.eye(new_mean.size(0), device=new_mean.device)
                new_stats[class_id] = GaussianStatistics(new_mean, new_cov)
        return new_stats

    def optimize_nonlinear_transform(self, features_before, features_after):
        features_after = F.normalize(features_after, dim=1)
        features_before = F.normalize(features_before, dim=1)
        model = NonlinearCompensator(features_after.size(1)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        steps = 5000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
        for step in range(steps):
            random_indices = torch.randint(0, features_before.shape[0], (32, ))
            features_before_batch = features_before[random_indices]
            features_after_batch = features_after[random_indices]
            optimizer.zero_grad()
            output = model(features_before_batch)
            loss = criterion(output, features_after_batch)
            loss_reg = model.reg_loss()
            loss_total = loss + self.gamma_2 * loss_reg
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 2500 == 0:
                print(f"弱非线性补偿器训练: Step {step + 1}, Loss_total: {loss_total.item():.4f}, Loss: {loss.item():.4f}, Loss_reg: {loss_reg.item():.4f}")
        model.eval()
        return model, tuple(F.softmax(model.alphas, dim=0).detach().cpu().tolist())
    
    def update_statistics_with_mlp_transform(self, stats, features_before, features_after):
        """使用纯MLP进行特征变换"""
        print("基于当前任务的前后特征构建MLP补偿器")
        features_before = features_before.to(self.device)
        features_after = features_after.to(self.device)
        mapping = self.optimize_mlp_transform(features_before, features_after)
        with torch.no_grad():
            feats_new_after_pred = mapping(features_before)
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()
        print(f"MLP补偿后特征差异: {feat_diffs_pred:.4f}")
        
        with torch.no_grad():
            new_stats = {}
            for class_id, gauss in stats.items():
                class_mean = gauss.mean.to(self.device)
                class_L = gauss.L.to(self.device)
                d = class_mean.size(0)
                num = d * 10
                class_samples = sample_torch_cholesky(num, class_mean, class_L, self.cached_Z)
                class_samples = mapping(class_samples)
                new_mean = class_samples.mean(dim=0)
                new_cov = torch.cov(class_samples.T) + 1e-3 * torch.eye(new_mean.size(0), device=new_mean.device)
                new_stats[class_id] = GaussianStatistics(new_mean, new_cov)
        return new_stats

    def optimize_mlp_transform(self, features_before, features_after):
        """训练纯MLP补偿器"""
        class MLPCompensator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MLPCompensator, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, input_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(input_dim, output_dim, bias=True))
            def forward(self, x):
                return self.fc(x)

        model = MLPCompensator(features_before.shape[1], features_after.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        criterion = nn.MSELoss()
        steps = 5000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
        for step in range(steps):
            random_indices = torch.randint(0, features_before.shape[0], (32, ))
            features_before_batch = features_before[random_indices]
            features_after_batch = features_after[random_indices]
            optimizer.zero_grad()
            output = model(features_before_batch)
            loss = criterion(output, features_after_batch)
            loss.backward()
            optimizer.step()
            if (step + 1) % 1000 == 0:
                print(f"MLP补偿器训练: Step {step + 1}, Loss: {loss.item():.4f}")
            scheduler.step()
        model.eval()
        return model
    

    def refine_classifiers(self, fc, task_id):
        if task_id > 0:
            # 不使用辅助数据的分类器
            original_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.original_stats)
            linear_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.linear_stats) if self.use_linear_compensation else None
            weak_nonlinear_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.weak_nonlinear_stats) if self.use_weak_nonlinear_compensation else None
            mlp_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.mlp_stats) if self.use_mlp_compensation else None
            
            # 使用辅助数据的分类器
            if self.use_auxiliary_data:
                linear_fc_with_aux = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.linear_stats_with_aux) if self.use_linear_compensation else None
                weak_nonlinear_fc_with_aux = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.weak_nonlinear_stats_with_aux) if self.use_weak_nonlinear_compensation else None
                mlp_fc_with_aux = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.mlp_stats_with_aux) if self.use_mlp_compensation else None
            else:
                linear_fc_with_aux = None
                weak_nonlinear_fc_with_aux = None
                mlp_fc_with_aux = None
                
        elif task_id == 0:
            # 初始任务，直接复制分类器
            original_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.original_stats)
            linear_fc = copy.deepcopy(original_fc) if self.use_linear_compensation else None
            weak_nonlinear_fc = copy.deepcopy(original_fc) if self.use_weak_nonlinear_compensation else None
            mlp_fc = copy.deepcopy(original_fc) if self.use_mlp_compensation else None
            
            if self.use_auxiliary_data:
                linear_fc_with_aux = copy.deepcopy(original_fc) if self.use_linear_compensation else None
                weak_nonlinear_fc_with_aux = copy.deepcopy(original_fc) if self.use_weak_nonlinear_compensation else None
                mlp_fc_with_aux = copy.deepcopy(original_fc) if self.use_mlp_compensation else None
            else:
                linear_fc_with_aux = None
                weak_nonlinear_fc_with_aux = None
                mlp_fc_with_aux = None
        
        return {
            'without_aux': {
                'original': original_fc,
                'linear': linear_fc,
                'weak_nonlinear': weak_nonlinear_fc,
                'mlp': mlp_fc
            },
            'with_aux': {
                'linear': linear_fc_with_aux,
                'weak_nonlinear': weak_nonlinear_fc_with_aux,
                'mlp': mlp_fc_with_aux
            }
        }

        
    def train_classifier_with_cached_samples(self, fc, stats):
        epochs = 6
        num_samples_per_class = 1024
        batch_size = 32 * len(stats) // 10
        lr = 5e-4     
        fc.to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr/10)
        
        cached_Z = self.cached_Z

        # Prepare distribution parameters
        dists = []
        labels = []
        for class_id, gauss in stats.items():
            class_mean = gauss.mean.to(self.device)
            class_L = gauss.L.to(self.device)
            dists.append((class_mean, class_L))
            labels.append(class_id)
        
        fc.train()
        samples, targets = [], []

        # Generate samples using cached Z - shuffle before each training run
        cached_Z = cached_Z.to(self.device)
        shuffle_idx = torch.randperm(cached_Z.size(0), device=self.device)
        cached_Z = cached_Z[shuffle_idx]
        
        for class_id, (mean, L) in zip(labels, dists):
            # Calculate start and end indices with modulo to handle overflow
            start_idx = (class_id * num_samples_per_class) % cached_Z.size(0)
            end_idx = start_idx + num_samples_per_class
            
            if end_idx <= cached_Z.size(0):
                Z = cached_Z[start_idx:end_idx]
            else:
                # Handle overflow by wrapping around
                remaining = end_idx - cached_Z.size(0)
                Z = torch.cat([cached_Z[start_idx:], cached_Z[:remaining]], dim=0)
            
            # Generate samples: X = μ + LZ
            samples.append(mean + torch.mm(Z, L.t()))
            targets.append(torch.full((num_samples_per_class,), class_id, device=self.device))

        inputs = torch.cat(samples, dim=0)
        targets = torch.cat(targets, dim=0)

        # Training loop
        for epoch in range(epochs):
            sf_indexes = torch.randperm(inputs.size(0), device=self.device)
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            num_samples = inputs.size(0)
            num_complete_batches = num_samples // batch_size
            
            for batch_idx in range(num_complete_batches + 1):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)                
                if start_idx >= end_idx:
                    continue                    
                inp = inputs[start_idx:end_idx]
                tgt = targets[start_idx:end_idx]               
                optimizer.zero_grad()
                output = fc(inp)
                loss = symmetric_cross_entropy_loss(output, tgt)
                loss.backward()
                optimizer.step()
                losses += loss.item() * (end_idx - start_idx)
            loss = losses / num_samples
            if (epoch + 1) % 3 == 0:
                print(f"分类器矫正训练 (cached samples): Epoch {epoch + 1}, Loss: {loss:.4f}")
            scheduler.step() 
        return fc

    def get_aux_loader(self, args):
       
       
        if hasattr(self, 'loader'):
            return self.loader

       
        aux_dataset_type = args.get('aux_dataset_type', 'image_folder')
        num_samples = args.get('auxiliary_data_size', 1024)
        batch_size = args.get('batch_size', 128)
        num_workers = args.get('num_workers', 4)

        
        transform = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 归一化参数
        ])

           # 根据 aux_dataset_type 加载数据集
        if aux_dataset_type == 'image_folder':
          if 'auxiliary_data_path' not in args:
            raise ValueError("当 aux_dataset_type='image_folder' 时，必须提供 auxiliary_data_path")
          dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        elif aux_dataset_type == 'cifar10':
          dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
          dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        else:
          raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

         # 随机采样指定数量的样本
        torch.manual_seed(1)  # 固定随机种子以保证可重复性
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        train_subset = Subset(dataset, indices)

        # 创建 DataLoader 并缓存
        self.loader = DataLoader(train_subset, batch_size=128, shuffle=True, 
                             num_workers=4, pin_memory=True)
        return self.loader




'''
    def get_aux_loader(self, args):
        try:
            return self.aux_loader
        except:
            transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
            num_samples = args['auxiliary_data_size']
            torch.manual_seed(1)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            train_subset = Subset(dataset, indices)
            self.loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            return self.loader
    
'''

