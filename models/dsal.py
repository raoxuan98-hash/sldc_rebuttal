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
            matrix.size(0), new_cols, device=matrix.device, dtype=matrix.dtype
        )
    padding = torch.zeros(
        matrix.size(0), new_cols, device=matrix.device, dtype=matrix.dtype
    )
    return torch.cat([matrix, padding], dim=1)


class DSALLearner(BaseLearner):
    """Dual-Stream Analytic Learner (DS-AL) — fixed version."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FinetuneIncrementalNet(
            args, pretrained=True)
        self._network.to(self._device)
        self._network.eval()
        for param in self._network.convnet.parameters():
            param.requires_grad = False

        # --- 使用双精度，保持与论文一致 ---
        self._dtype = torch.double

        # --- 形状与超参 ---
        self._feature_dim = self._network.feature_dim
        self._rf_dim = int(args.get("random_feature_dim", 8192))

        # 支持主/补偿分开正则（映射论文 gamma_main / gamma_comp）
        ridge_main = float(args.get("ridge_lambda_main", args.get("ridge_lambda", 1e-3)))
        ridge_comp = float(args.get("ridge_lambda_comp", args.get("ridge_lambda", 1e-3)))
        self._ridge_lambda_main = ridge_main
        self._ridge_lambda_comp = ridge_comp

        # RLS 的数值稳定项
        self._rls_eps = float(args.get("rls_eps", 1e-6))

        # 融合权重 C
        self._fusion_weight = float(args.get("dsal_fusion_weight", 1.0))

        # 激活函数：对齐官方默认 main=relu, comp=tanh
        activation_main = args.get("dsal_main_activation", "relu")
        activation_comp = args.get("dsal_comp_activation", "tanh")
        self._activation_main = _activation(activation_main)
        self._activation_comp = _activation(activation_comp)

        # ============= 关键修复 1：随机投影尺度（加入 1/sqrt(d) 缩放，防止 tanh 饱和） =============
        scale = 1.0 / math.sqrt(self._feature_dim)
        self._weight_random = torch.randn(
            self._feature_dim, self._rf_dim, device=self._device, dtype=self._dtype
        ) * scale
        self._bias_random = torch.zeros(
            self._rf_dim, device=self._device, dtype=self._dtype
        )

        # R 矩阵与权重矩阵分别初始化（主/补偿可不同正则）
        I_main = torch.eye(self._rf_dim, device=self._device, dtype=self._dtype)
        I_comp = torch.eye(self._rf_dim, device=self._device, dtype=self._dtype)
        self._R_main = I_main / self._ridge_lambda_main
        self._R_comp = I_comp / self._ridge_lambda_comp

        self._W_main = torch.zeros(self._rf_dim, 0, device=self._device, dtype=self._dtype)
        self._W_comp = torch.zeros(self._rf_dim, 0, device=self._device, dtype=self._dtype)

        # FSA（按题意忽略；保留开关以兼容你的框架）
        self._fsa_done = False
        self._use_fsa = bool(args.get("first_section_adaptation", True))
        self._fsa_steps = int(args.get("fsa_steps", 1000))
        self._fsa_lr = float(args.get("fsa_lr", 1e-4))
        self._fsa_wd = float(args.get("fsa_weight_decay", 3e-5))
        self._fsa_bs = int(args.get("fsa_batch_size", 16))

    # ------------------------------------------------------------------
    # Feature helpers (double)
    # ------------------------------------------------------------------
    def _project(self, features: torch.Tensor, stream: str) -> torch.Tensor:
        if features.dtype != self._dtype:
            features = features.to(self._dtype)
        projected = features @ self._weight_random + self._bias_random
        if stream == "main":
            return self._activation_main(projected)
        return self._activation_comp(projected)

    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        # pin_memory_device="cuda" 需要 torch>=2.0 且设备为 CUDA；你的环境若报错可去掉该参数
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=shuffle,
        )

    @torch.inference_mode()
    def _extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        self._network.eval()
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)  # keep on GPU
            features = self._network.convnet(inputs)["features"]  # float32 on GPU
            feats.append(features)  # keep on GPU
            labels.append(targets)
        features_all = torch.cat(feats, dim=0)  # still on GPU
        labels_all = torch.cat(labels, dim=0)   # still on GPU
        return {"features": features_all, "labels": labels_all}

    def _one_hot(self, labels: torch.Tensor, total_classes: int) -> torch.Tensor:
        one_hot = torch.zeros(
            labels.size(0), total_classes, device=self._device, dtype=self._dtype
        )
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return one_hot

    # ------------------------------------------------------------------
    # （可选）FSA — 题意不关注，保持原样以兼容
    # ------------------------------------------------------------------
    def _first_section_adaptation(self, data_manager) -> None:
        first_task_size = data_manager.get_task_size(0)
        train_dataset = data_manager.get_dataset(
            np.arange(0, first_task_size),
            source="train",
            mode="train",
            appendent=[],
            with_raw=False,
        )
        train_loader = self._build_dataloader(
            train_dataset, batch_size=16, shuffle=True
        )

        for n, p in self._network.convnet.named_parameters():
            if "A" in n or "B" in n or "norm" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self._network.train()

        classifier = torch.nn.Linear(self._feature_dim, first_task_size).to(self._device)
        optimizer = torch.optim.AdamW(
            list(self._network.convnet.parameters()) + list(classifier.parameters()),
            lr=self._fsa_lr,
            weight_decay=self._fsa_wd,
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        data_iter = itertools.cycle(train_loader)
        ema_beta = 0.95
        ema_loss = None
        ema_acc = None

        steps_done = 0
        while steps_done < self._fsa_steps:
            _, inputs, targets = next(data_iter)
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            features = self._network.convnet(inputs)["features"]
            logits = classifier(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == targets).float().mean().item()

            loss_item = loss.item()
            if ema_loss is None:
                ema_loss = loss_item
                ema_acc = acc
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_item
                ema_acc = ema_beta * ema_acc + (1 - ema_beta) * acc

            steps_done += 1
            if steps_done % 50 == 0 or steps_done == self._fsa_steps:
                logging.info(
                    "[FSA] step=%d/%d | loss_ema=%.4f | acc_ema=%.2f%%",
                    steps_done, self._fsa_steps, ema_loss, ema_acc * 100.0,
                )

        del classifier
        torch.cuda.empty_cache()
        for p in self._network.convnet.parameters():
            p.requires_grad = False
        self._network.eval()
        self._fsa_done = True
        logging.info("[FSA] first_section_adaptation completed; convnet re-frozen.")

    # ------------------------------------------------------------------
    # RLS with double precision
    # ------------------------------------------------------------------
    def _compute_gain(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if X.dtype != self._dtype:
            X = X.to(self._dtype)
        XR = X @ R
        # K = I + X R X^T（加入 eps 做数值稳定）
        S = XR @ X.t()
        eye = torch.eye(S.size(0), device=S.device, dtype=S.dtype)
        S = S + eye + self._rls_eps * eye
        # 求解 K^T G^T = (XR)^T  =>  G = (solve(S^T, XR))^T
        gain_t = torch.linalg.solve(S.transpose(-1, -2), XR)
        return gain_t.transpose(0, 1)

    def _update_main(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        if X.dtype != self._dtype:
            X = X.to(self._dtype)
        if Y.dtype != self._dtype:
            Y = Y.to(self._dtype)
        W_prev = _pad_columns(self._W_main, Y.size(1) - self._W_main.size(1))
        gain = self._compute_gain(X, self._R_main)
        self._R_main = self._R_main - gain @ X @ self._R_main
        residual = Y - X @ W_prev
        self._W_main = W_prev + gain @ residual

    # ============= 关键修复 2：补偿流做“全列 RLS + PLC 清洗” =============
    def _update_compensation(
        self, X: torch.Tensor, residual_new: torch.Tensor, new_class_count: int
    ) -> None:
        if X.dtype != self._dtype:
            X = X.to(self._dtype)
        if residual_new.dtype != self._dtype:
            residual_new = residual_new.to(self._dtype)

        # 1) 先把 W_comp 扩到新类别总数
        W_prev_full = _pad_columns(self._W_comp, new_class_count)

        # 2) 构造“全列”目标：旧类=0，新类=residual_new（Previous Label Cleansing）
        total_cols = W_prev_full.size(1)
        Y_comp_full = torch.zeros(
            X.size(0), total_cols, device=self._device, dtype=self._dtype
        )
        # 只在新类列填 residual_new
        Y_comp_full[:, self._known_classes:self._total_classes] = residual_new

        # 3) 标准 RLS 更新（与主流相同，但用 R_comp）
        gain = self._compute_gain(X, self._R_comp)
        self._R_comp = self._R_comp - gain @ X @ self._R_comp
        residual_full = Y_comp_full - X @ W_prev_full
        self._W_comp = W_prev_full + gain @ residual_full

    # ------------------------------------------------------------------
    # Incremental & Eval (double)
    # ------------------------------------------------------------------
    def incremental_train(self, data_manager):
        """Class-balanced RLS training for DS-AL."""
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self._total_classes, 5)

        # --- 1. 构建当前任务数据集 ---
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=[],
            with_raw=False,
        )
        self.train_loader = self._build_dataloader(
            train_dataset, batch_size=64, shuffle=False
        )

        # --- 2. 提取特征与标签 ---
        batch_data = self._extract_features(self.train_loader)
        features = batch_data["features"]  # [N, D]
        labels = batch_data["labels"]      # [N]

        # ================================================================
        # 🧩 Step 3: Class-balanced weighting
        # ================================================================
        unique_labels, counts = torch.unique(labels, return_counts=True)
        count_dict = {int(k): float(v) for k, v in zip(unique_labels, counts)}

        # 每个样本的权重 w_i = 1 / sqrt(n_class)
        weights = torch.tensor(
            [1.0 / math.sqrt(count_dict[int(lbl)]) for lbl in labels],
            device=self._device,
            dtype=self._dtype,
        ).view(-1, 1)

        # ================================================================
        # 🧩 Step 4: 主流更新（Main stream RLS）
        # ================================================================
        X_main = self._project(features, stream="main") * weights  # [N, rf_dim]
        targets = self._one_hot(labels, self._total_classes) * weights  # [N, C]
        self._update_main(X_main, targets)

        # --- 计算残差 ---
        with torch.no_grad():
            main_logits = X_main @ self._W_main
        residual_full = targets - main_logits
        residual_new = residual_full[:, self._known_classes : self._total_classes]

        # ================================================================
        # 🧩 Step 5: 补偿流更新（Compensation stream RLS）
        # ================================================================
        X_comp = self._project(features, stream="comp") * weights
        residual_new = residual_new * weights
        self._update_compensation(X_comp, residual_new, task_size)

        # --- 6. 更新状态 ---
        self._known_classes = self._total_classes

    def _forward_main(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"].to(self._dtype)
        X = self._project(features, stream="main")
        return X @ self._W_main

    def _forward_comp(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"].to(self._dtype)
        X = self._project(features, stream="comp")
        return X @ self._W_comp

    def eval_task(self):
        correct_main = 0
        correct_fused = 0
        total = 0

        C = self._fusion_weight

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                logits_main = self._forward_main(inputs)
                logits_comp = self._forward_comp(inputs)
                logits_fused = logits_main + C * logits_comp

                preds_main = torch.argmax(logits_main, dim=1)
                preds_fused = torch.argmax(logits_fused, dim=1)

                correct_main += (preds_main == targets).sum().item()
                correct_fused += (preds_fused == targets).sum().item()
                total += targets.size(0)

        acc_main = 100.0 * correct_main / max(total, 1)
        acc_fused = 100.0 * correct_fused / max(total, 1)
        logging.info(
            "Task %d | DS-AL Main: %.2f%% | DS-AL Fused: %.2f%% (C=%.3f)",
            self._cur_task, acc_main, acc_fused, C,
        )
        return {
            "dsal_main": round(acc_main, 2),
            "dsal_fused": round(acc_fused, 2),
        }

    def loop(self, data_manager):
        final_results: Dict[str, List[float]] = {
            "dsal_main": [],
            "dsal_fused": [],
        }

        # 题意：FSA 不用管；若你希望完全禁用，将 _use_fsa 设为 False 即可
        if self._use_fsa and not self._fsa_done:
            self._first_section_adaptation(data_manager)

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