import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Dict
from timm.models.vision_transformer import VisionTransformer as timm_ViT

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int):
        super().__init__()
        assert r > 0
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        # LoRA 参数
        self.A = nn.Parameter(torch.zeros(r, self.in_features, dtype=linear.weight.dtype, device=linear.weight.device))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=linear.weight.dtype, device=linear.weight.device))

        # 初始化：A Kaiming，B 全零 → 初始恒等
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # 冻结原始层
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.linear(x)
        h = F.linear(x, self.A)          # (..., r)
        lora = F.linear(h, self.B)       # (..., out_features)
        return orig + lora

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        delta = self.B @ self.A          # (out, in)
        self.linear.weight.add_(delta.to(self.linear.weight.dtype))
        self.B.zero_()

    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        W = self.linear.weight
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.copy_(Vh[: self.r, :])
        self.B.zero_()


# -----------------------------
# QKV 专用 LoRA 适配器 —— 无 scale
# -----------------------------
class LoRAQKV(nn.Module):
    """
    ViT 中的 qkv 层 LoRA：W_qkv <- W_qkv + (B @ A)
    """
    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        assert r > 0
        self.qkv = qkv
        self.dim = qkv.in_features
        assert qkv.out_features % 3 == 0 and qkv.out_features == 3 * self.dim, \
            "Expect qkv.out_features == 3 * qkv.in_features for ViT."

        self.r = r

        self.A = nn.Parameter(torch.zeros(r, self.dim, dtype=qkv.weight.dtype, device=qkv.weight.device))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r, dtype=qkv.weight.dtype, device=qkv.weight.device))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.qkv.weight.requires_grad_(False)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.qkv(x)
        h = F.linear(x, self.A)          # (..., r)
        lora = F.linear(h, self.B)       # (..., 3*dim)
        return orig + lora

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        delta = self.B @ self.A          # (3*dim, dim)
        self.qkv.weight.add_(delta.to(self.qkv.weight.dtype))
        self.B.zero_()

    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        W = self.qkv.weight
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.copy_(Vh[: self.r, :])
        self.B.zero_()


# -----------------------------
# 主包装器：PlainLoRAViT —— 无 alpha，无 scale
# -----------------------------
class PlainLoRAViT(nn.Module):
    """
    仅含 LoRA(A,B) 的 ViT 包装器：
      - 替换指定 block 的 attn.qkv、attn.proj、mlp.fc1、mlp.fc2
      - 冻结原始 ViT，仅训练 A、B
      - 支持 SVD 初始化、权重合并、去 LoRA 化
    """
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,  # 默认全部 block,
        include_norm: bool = True
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"

        self.r = r

        # 默认所有 block
        self.lora_layer = (
            list(lora_layer) if lora_layer is not None
            else list(range(len(vit_model.blocks)))
        )

        # 冻结 ViT 原始参数
        for n, p in vit_model.named_parameters():
            if include_norm and "norm" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad = False

        # 替换模块
        self.lora_modules = nn.ModuleDict()
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # --- QKV ---
            qkv_adapter = LoRAQKV(blk.attn.qkv, r=self.r)
            blk.attn.qkv = qkv_adapter
            self.lora_modules[f"block_{idx}_attn_qkv"] = qkv_adapter

            # --- Attention Projection ---
            proj_adapter = LoRALinear(blk.attn.proj, r=self.r)
            blk.attn.proj = proj_adapter
            self.lora_modules[f"block_{idx}_attn_proj"] = proj_adapter

            # --- MLP fc1 ---
            fc1_adapter = LoRALinear(blk.mlp.fc1, r=self.r)
            blk.mlp.fc1 = fc1_adapter
            self.lora_modules[f"block_{idx}_mlp_fc1"] = fc1_adapter


            # --- MLP fc2 ---
            fc2_adapter = LoRALinear(blk.mlp.fc2, r=self.r)
            blk.mlp.fc2 = fc2_adapter
            self.lora_modules[f"block_{idx}_mlp_fc2"] = fc2_adapter

        self.vit = vit_model

        # SVD 初始化
        self.reset_parameters_svd()
        self.feature_dim = vit_model.embed_dim
        self.embed_dim = vit_model.embed_dim
        self.optimizable = False
        self.use_projection = False

    # ---------- 前向传播 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.vit(x)
        return {"features": feature}

    # ---------- 工具方法 ----------
    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        for mod in self.lora_modules.values():
            mod.reset_parameters_svd()

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for mod in self.lora_modules.values():
            yield mod.A
            yield mod.B

    def kl_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        self.eval()
        for mod in self.lora_modules.values():
            mod.merge_lora_weights()

    @torch.no_grad()
    def finalize_without_lora(self) -> None:
        """
        1) 合并 LoRA 权重；
        2) 移除适配器，恢复原生 Linear；
        3) 清空 lora_modules。
        """
        self.merge_lora_weights()

        for idx, blk in enumerate(self.vit.blocks):
            name_qkv = f"block_{idx}_attn_qkv"
            if name_qkv in self.lora_modules:
                adapter = self.lora_modules[name_qkv]
                blk.attn.qkv = adapter.qkv

            name_proj = f"block_{idx}_attn_proj"
            if name_proj in self.lora_modules:
                adapter = self.lora_modules[name_proj]
                blk.attn.proj = adapter.linear

            name_fc1 = f"block_{idx}_mlp_fc1"
            if name_fc1 in self.lora_modules:
                adapter = self.lora_modules[name_fc1]
                blk.mlp.fc1 = adapter.linear

            name_fc2 = f"block_{idx}_mlp_fc2"
            if name_fc2 in self.lora_modules:
                adapter: LoRALinear = self.lora_modules[name_fc2]
                blk.mlp.fc2 = adapter.linear

        self.lora_modules = nn.ModuleDict()
