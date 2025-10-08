import logging
import math
from typing import Dict, List

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
    """Dual-Stream Analytic Learner (DS-AL).

    Extends ACIL with a compensation stream that fits residuals for the new
    classes using a different activation function.
    """

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

        self._feature_dim = self._network.feature_dim
        self._rf_dim = args.get("random_feature_dim", 8192)
        self._ridge_lambda = args.get("ridge_lambda", 1e-3)
        self._rls_eps = args.get("rls_eps", 1e-6)
        self._fusion_weight = args.get("dsal_fusion_weight", 1.0)

        activation_main = args.get("dsal_main_activation", args.get("acil_activation", "gelu"))
        activation_comp = args.get("dsal_comp_activation", "tanh")
        self._activation_main = _activation(activation_main)
        self._activation_comp = _activation(activation_comp)

        self._weight_random = (
            torch.randn(
                self._feature_dim,
                self._rf_dim,
                device=self._device,
                dtype=torch.float32,
            )
            / math.sqrt(self._feature_dim)
        )
        self._bias_random = torch.zeros(
            self._rf_dim, device=self._device, dtype=torch.float32
        )

        identity = torch.eye(
            self._rf_dim, device=self._device, dtype=torch.float32
        )
        self._R_main = identity / self._ridge_lambda
        self._R_comp = identity.clone() / self._ridge_lambda
        self._W_main = torch.zeros(
            self._rf_dim, 0, device=self._device, dtype=torch.float32
        )
        self._W_comp = torch.zeros(
            self._rf_dim, 0, device=self._device, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _project(self, features: torch.Tensor, stream: str) -> torch.Tensor:
        projected = features @ self._weight_random + self._bias_random
        if stream == "main":
            return self._activation_main(projected)
        return self._activation_comp(projected)

    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.get("num_workers", 4),
        )

    def _extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        self._network.eval()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device, non_blocking=True)
                features = self._network.convnet(inputs)["features"]
                feats.append(features.cpu())
                labels.append(targets)
        features_all = torch.cat(feats, dim=0).to(self._device)
        labels_all = torch.cat(labels, dim=0).to(self._device)
        return {"features": features_all, "labels": labels_all}

    def _one_hot(self, labels: torch.Tensor, total_classes: int) -> torch.Tensor:
        one_hot = torch.zeros(
            labels.size(0), total_classes, device=self._device, dtype=torch.float32
        )
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return one_hot

    # ------------------------------------------------------------------
    # Recursive least squares updates
    # ------------------------------------------------------------------
    def _compute_gain(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        XR = X @ R
        S = XR @ X.t()
        eye = torch.eye(S.size(0), device=S.device, dtype=S.dtype)
        S = S + eye + self._rls_eps * eye
        gain_t = torch.linalg.solve(S.transpose(-1, -2), XR)
        return gain_t.transpose(0, 1)

    def _update_main(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        W_prev = _pad_columns(self._W_main, Y.size(1) - self._W_main.size(1))
        gain = self._compute_gain(X, self._R_main)
        self._R_main = self._R_main - gain @ X @ self._R_main
        residual = Y - X @ W_prev
        self._W_main = W_prev + gain @ residual

    def _update_compensation(
        self, X: torch.Tensor, residual_new: torch.Tensor, new_class_count: int
    ) -> None:
        W_prev_full = _pad_columns(self._W_comp, new_class_count)
        current_block = W_prev_full[:, -new_class_count:].clone()
        gain = self._compute_gain(X, self._R_comp)
        self._R_comp = self._R_comp - gain @ X @ self._R_comp
        block_residual = residual_new - X @ current_block
        updated_block = current_block + gain @ block_residual
        W_prev_full[:, -new_class_count:] = updated_block
        self._W_comp = W_prev_full

    # ------------------------------------------------------------------
    # Incremental routine
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
            with_raw=False,
        )
        self.train_loader = self._build_dataloader(
            train_dataset, self.args.get("batch_size", 64), shuffle=False
        )

        batch_data = self._extract_features(self.train_loader)
        features = batch_data["features"]
        labels = batch_data["labels"]

        X_main = self._project(features, stream="main")
        targets = self._one_hot(labels, self._total_classes)
        self._update_main(X_main, targets)

        with torch.no_grad():
            main_logits = X_main @ self._W_main
        residual_full = targets - main_logits
        residual_new = residual_full[:, self._known_classes : self._total_classes]

        X_comp = self._project(features, stream="comp")
        self._update_compensation(X_comp, residual_new, task_size)

        self._known_classes = self._total_classes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _forward_main(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"]
        X = self._project(features, stream="main")
        return X @ self._W_main

    def _forward_comp(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"]
        X = self._project(features, stream="comp")
        return X @ self._W_comp

    def eval_task(self):
        correct_main = 0
        correct_fused = 0
        total = 0

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                logits_main = self._forward_main(inputs)
                logits_comp = self._forward_comp(inputs)
                logits_fused = logits_main + self._fusion_weight * logits_comp

                preds_main = torch.argmax(logits_main, dim=1)
                preds_fused = torch.argmax(logits_fused, dim=1)

                correct_main += (preds_main == targets).sum().item()
                correct_fused += (preds_fused == targets).sum().item()
                total += targets.size(0)

        acc_main = 100.0 * correct_main / max(total, 1)
        acc_fused = 100.0 * correct_fused / max(total, 1)
        logging.info(
            "Task %d | DS-AL Main: %.2f%% | DS-AL Fused: %.2f%%",
            self._cur_task,
            acc_main,
            acc_fused,
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

        for task in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)

            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes), source="test", mode="test"
            )
            self.test_loader = self._build_dataloader(
                test_dataset, self.args.get("batch_size", 64), shuffle=False
            )

            task_results = self.eval_task()
            for key, value in task_results.items():
                final_results.setdefault(key, [])
                final_results[key].append(value)

            super().after_task()

        return final_results
