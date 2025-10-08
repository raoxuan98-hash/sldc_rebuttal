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
    """Analytical Continual Incremental Learning (ACIL).

    This learner freezes the ViT backbone and performs incremental ridge-regression
    updates in a random feature space using the recursive least squares (RLS)
    formulation described in the accompanying documentation.
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
        activation_name = args.get("acil_activation", "gelu")
        self._activation = _activation(activation_name)

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
        self._R = identity / self._ridge_lambda
        self._W = torch.zeros(
            self._rf_dim, 0, device=self._device, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Core mathematics
    # ------------------------------------------------------------------
    def _compute_random_features(self, z: torch.Tensor) -> torch.Tensor:
        projected = z @ self._weight_random + self._bias_random
        return self._activation(projected)

    def _rls_gain(self, X: torch.Tensor) -> torch.Tensor:
        XR = X @ self._R
        S = XR @ X.t()
        batch_eye = torch.eye(
            S.size(0), device=S.device, dtype=S.dtype
        )
        S = S + batch_eye + self._rls_eps * batch_eye
        gain_t = torch.linalg.solve(S.transpose(-1, -2), XR)
        return gain_t.transpose(0, 1)

    def _rls_update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
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
            num_workers=self.args.get("num_workers", 4),
        )

    def _extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._network.eval()
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
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
            with_raw=False,
        )

        self.train_loader = self._build_dataloader(
            train_dataset, self.args.get("batch_size", 64), shuffle=False
        )

        batch_data = self._extract_features(self.train_loader)
        features = batch_data["features"]
        labels = batch_data["labels"]

        X_k = self._compute_random_features(features)
        Y_k = self._one_hot(labels, self._total_classes)
        self._rls_update(X_k, Y_k)

        self._known_classes = self._total_classes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._network.convnet(inputs)["features"]
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

    def loop(self, data_manager):
        final_results = {"acil": []}
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
