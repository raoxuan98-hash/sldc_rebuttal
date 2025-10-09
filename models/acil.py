import logging
import math
from typing import Dict, List
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import FinetuneIncrementalNet


def _activation(name: str):
    name = name.lower()
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "tanh":
        return torch.tanh
    if name == "mish":
        return torch.nn.functional.mish
    raise ValueError(f"Unsupported activation: {name}")


def _pad_columns(matrix: torch.Tensor, new_cols: int) -> torch.Tensor:
    if new_cols <= 0:
        return matrix
    if matrix.numel() == 0:
        return torch.zeros(
            matrix.size(0),
            new_cols,
            device=matrix.device,
            dtype=matrix.dtype,
        )
    padding = torch.zeros(
        matrix.size(0), new_cols, device=matrix.device, dtype=matrix.dtype
    )
    return torch.cat([matrix, padding], dim=1)


class ACILLearner(BaseLearner):
    """Analytical Continual Incremental Learning (ACIL) — no scaling, dtype=torch.double."""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FinetuneIncrementalNet(
            args, pretrained=True, num_used_layers=args.get("num_used_layers", 1)
        )
        self._network.to(self._device)
        self._network.eval()
        for param in self._network.convnet.parameters():
            param.requires_grad = False

        # --- dtype 全链路改为 double ---
        self._dtype = torch.double

        self._feature_dim = self._network.feature_dim
        self._rf_dim = args.get("random_feature_dim", 8192)
        self._ridge_lambda = args.get("ridge_lambda", 1e-3)
        self._rls_eps = args.get("rls_eps", 1e-6)
        activation_name = args.get("acil_activation", "gelu")
        self._activation = _activation(activation_name)

        # --- 不再做 1/sqrt(d) 缩放；保持随机权重固定为 buffer (非可训练) 语义 ---
        self._weight_random = torch.randn(
            self._feature_dim,
            self._rf_dim,
            device=self._device,
            dtype=self._dtype,
        )
        self._bias_random = torch.zeros(
            self._rf_dim, device=self._device, dtype=self._dtype
        )

        identity = torch.eye(self._rf_dim, device=self._device, dtype=self._dtype)
        self._R = identity / self._ridge_lambda
        self._W = torch.zeros(self._rf_dim, 0, device=self._device, dtype=self._dtype)

        self._fsa_done = False
        self._use_fsa = bool(args.get("first_section_adaptation", True))
        self._fsa_steps = int(args.get("fsa_steps", 1000))
        self._fsa_lr = float(args.get("fsa_lr", 1e-4))
        self._fsa_wd = float(args.get("fsa_weight_decay", 0.0))
        self._fsa_bs = int(args.get("fsa_batch_size", 64))

    # ------------------------------------------------------------------
    # Core mathematics
    # ------------------------------------------------------------------
    def _compute_random_features(self, z: torch.Tensor) -> torch.Tensor:
        # 确保特征为 double
        if z.dtype != self._dtype:
            z = z.to(dtype=self._dtype)
        projected = z @ self._weight_random + self._bias_random
        return self._activation(projected)

    def _rls_gain(self, X: torch.Tensor) -> torch.Tensor:
        if X.dtype != self._dtype:
            X = X.to(dtype=self._dtype)
        XR = X @ self._R
        S = XR @ X.t()
        batch_eye = torch.eye(S.size(0), device=S.device, dtype=S.dtype)
        S = S + batch_eye + self._rls_eps * batch_eye
        gain_t = torch.linalg.solve(S.transpose(-1, -2), XR)
        return gain_t.transpose(0, 1)

    def _rls_update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        if X.dtype != self._dtype:
            X = X.to(dtype=self._dtype)
        if Y.dtype != self._dtype:
            Y = Y.to(dtype=self._dtype)

        W_prev = _pad_columns(self._W, Y.size(1) - self._W.size(1))
        gain = self._rls_gain(X)
        self._R = self._R - gain @ X @ self._R
        residual = Y - X @ W_prev
        self._W = W_prev + gain @ residual

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=shuffle,  # 训练 True，可减少最后一批病态尺寸
        )
    

    @torch.inference_mode()
    def _extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._network.eval()
        feats, labels = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device, non_blocking=True)
            z = self._network.convnet(inputs)["features"]           # (N,d) float32
            feats.append(z.cpu())                                   # 立刻搬回 CPU，省显存
            labels.append(targets)
        z_all = torch.cat(feats, 0).to(self._device)                # 再次上卡
        # 映射一次到 RF，再转 double（下面给出 _compute_random_features 的优化）
        X_all = self._compute_random_features(z_all)                # -> double
        y_all = torch.cat(labels, 0).to(self._device)
        return {"features": z_all, "X": X_all, "labels": y_all}


    def _one_hot(self, labels: torch.Tensor, total_classes: int) -> torch.Tensor:
        one_hot = torch.zeros(
            labels.size(0), total_classes, device=self._device, dtype=self._dtype
        )
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return one_hot

    # ------------------------------------------------------------------
    # Incremental interface
    # ------------------------------------------------------------------
    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self._total_classes, 5)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=[],
            with_raw=False)

        self.train_loader = self._build_dataloader(
            train_dataset, batch_size=64, shuffle=True)

        batch_data = self._extract_features(self.train_loader)
        features = batch_data["features"]
        labels = batch_data["labels"]

        X_k = self._compute_random_features(features)
        Y_k = self._one_hot(labels, self._total_classes)
        self._rls_update(X_k, Y_k)
        self._known_classes = self._total_classes

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"].to(self._dtype)
        X = self._compute_random_features(features)
        return X @ self._W

    def eval_task(self):
        correct = 0
        total = 0
        self._network.eval()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits = self._forward(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        accuracy = 100.0 * correct / max(total, 1)
        logging.info(
            "Task %d | ACIL Accuracy: %.2f%%",
            self._cur_task,
            accuracy,
        )
        return {"acil": round(accuracy, 2)}


    def _first_section_adaptation(self, data_manager):
        """
        用第一个任务的数据对 convnet 做短暂监督微调，适配数据分布。
        仅执行一次；不会改 ACIL 的解析头与随机特征。
        改为按固定优化步数训练，并使用指数滑动平均（EMA）记录指标。
        """
        first_task_size = data_manager.get_task_size(0)

        train_dataset = data_manager.get_dataset(
            np.arange(0, first_task_size),
            source="train",
            mode="train",
            appendent=[],
            with_raw=False,
        )
        train_loader = self._build_dataloader(
            train_dataset, batch_size=32, shuffle=True
        )

        # 解冻部分参数
        for n, p in self._network.convnet.named_parameters():
            if "lora" in n or "norm" in n or "bias" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self._network.train()

        # 临时分类头
        clf = torch.nn.Linear(self._feature_dim, first_task_size).to(self._device)
        optimizer = torch.optim.AdamW(
            list(self._network.convnet.parameters()) + list(clf.parameters()),
            lr=self._fsa_lr,
            weight_decay=3e-5,
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        data_iter = itertools.cycle(train_loader)

        # EMA 参数（可调）
        ema_beta = 0.95  # 越接近1，平滑越强
        ema_loss = None
        ema_acc = None

        steps_done = 0
        while steps_done < self._fsa_steps:
            _, inputs, targets = next(data_iter)
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            feats = self._network.convnet(inputs)["features"]
            logits = clf(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # 计算当前 batch 的准确率
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == targets).float().mean().item()

            loss_item = loss.item()

            # 更新 EMA
            if ema_loss is None:
                ema_loss = loss_item
                ema_acc = acc
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_item
                ema_acc = ema_beta * ema_acc + (1 - ema_beta) * acc

            steps_done += 1

            # 每 N 步或最后一步输出日志
            if steps_done % 50 == 0 or steps_done == self._fsa_steps:
                logging.info(
                    "[FSA] step=%d/%d | loss_ema=%.4f | acc_ema=%.2f%%",
                    steps_done,
                    self._fsa_steps,
                    ema_loss,
                    ema_acc * 100.0,
                )

        # 清理 & 冻结
        del clf
        torch.cuda.empty_cache()
        for p in self._network.convnet.parameters():
            p.requires_grad = False
        self._network.eval()
        self._fsa_done = True
        logging.info("[FSA] first_section_adaptation 完成（%d steps），convnet 已重新冻结。", self._fsa_steps)
        
    def loop(self, data_manager):
        # 🔧 在进入增量循环前做一次首任务自适应
        if self._use_fsa and not self._fsa_done:
            self._first_section_adaptation(data_manager)

        final_results = {"acil": []}
        for task in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)

            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes), source="test", mode="test"
            )
            self.test_loader = self._build_dataloader(
                test_dataset, batch_size=64, shuffle=False
            )

            task_results = self.eval_task()
            for key, value in task_results.items():
                final_results.setdefault(key, [])
                final_results[key].append(value)

            super().after_task()

        return final_results