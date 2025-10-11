import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional
from timm.models.vision_transformer import VisionTransformer as timm_ViT


# ==================================================
#  Unified LoRA/DoRA Linear Adapter
# ==================================================
class UnifiedLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, use_dora: bool = False, lora_scale: float = 1.0):
        super().__init__()
        assert r > 0
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.use_dora = use_dora
        self.lora_scale = lora_scale

        # LoRA params
        self.A = nn.Parameter(torch.zeros(r, self.in_features, dtype=linear.weight.dtype, device=linear.weight.device))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=linear.weight.dtype, device=linear.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # DoRA components (if enabled)
        if use_dora:
            with torch.no_grad():
                w = linear.weight.data
                w_norm = w.norm(p=2, dim=1, keepdim=True) + 1e-8
                self.weight_directions = nn.Parameter(w / w_norm, requires_grad=False)
                self.magnitude = nn.Parameter(w_norm.clone(), requires_grad=True)
        else:
            self.register_parameter("weight_directions", None)
            self.register_parameter("magnitude", None)

        # Freeze original layer
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dora:
            delta_W = self.B @ self.A  # (3*dim, dim)
            adapted_weight = (self.weight_directions + delta_W) * self.magnitude
            return F.linear(x, adapted_weight, self.linear.bias)
        else:
            delta_out = self.lora_scale * F.linear(F.linear(x, self.A), self.B)
            return self.linear(x) + delta_out
        
    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        if self.use_dora:
            delta = self.B @ self.A
            self.weight_directions.add_(delta)
            self.B.zero_()
        else:
            delta = self.B @ self.A * self.lora_scale
            self.linear.weight.add_(delta)
            self.B.zero_()

    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        if self.use_dora:
            W = self.weight_directions
        else:
            W = self.linear.weight
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.copy_(Vh[: self.r, :])
        self.B.zero_()

# ==================================================
#  Unified QKV Adapter (LoRA + DoRA)
# ==================================================
class UnifiedQKVLoRA(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, use_dora: bool = False, lora_scale: float = 1.0):
        super().__init__()
        assert qkv.out_features == 3 * qkv.in_features
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.use_dora = use_dora
        self.lora_scale = lora_scale

        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if use_dora:
            with torch.no_grad():
                w = qkv.weight.data
                w_norm = w.norm(p=2, dim=1, keepdim=True) + 1e-8
                self.weight_directions = nn.Parameter(w / w_norm, requires_grad=False)
                self.magnitude = nn.Parameter(w_norm.clone(), requires_grad=True)
        else:
            self.register_parameter("weight_directions", None)
            self.register_parameter("magnitude", None)

        self.qkv.weight.requires_grad_(False)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dora:
            delta_W = self.B @ self.A  # (3*dim, dim)
            adapted_weight = (self.weight_directions + delta_W) * self.magnitude
            return F.linear(x, adapted_weight, self.qkv.bias)
        else:
            delta_out = self.lora_scale * F.linear(F.linear(x, self.A), self.B)
            return self.qkv(x) + delta_out

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        if self.use_dora:
            delta = self.B @ self.A
            self.weight_directions.add_(delta)
            self.B.zero_()
        else:
            delta = self.B @ self.A * self.lora_scale
            self.qkv.weight.add_(delta)
            self.B.zero_()


class LoRAViT(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, use_dora: bool = True, lora_scale: float = 1.0, lora_layer: Optional[Iterable[int]] = None):
        super().__init__()
        self.r = r
        self.use_dora = use_dora
        self.lora_scale = lora_scale
        self.vit = vit_model
        self.lora_modules = nn.ModuleDict()

        for n, p in vit_model.named_parameters():
            p.requires_grad_(False)

        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            blk.attn.qkv = UnifiedQKVLoRA(blk.attn.qkv, r, use_dora, lora_scale)
            blk.attn.proj = UnifiedLoRALinear(blk.attn.proj, r, use_dora, lora_scale)
            blk.mlp.fc1 = UnifiedLoRALinear(blk.mlp.fc1, r, use_dora, lora_scale)
            blk.mlp.fc2 = UnifiedLoRALinear(blk.mlp.fc2, r, use_dora, lora_scale)

            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            self.lora_modules[f"block_{idx}_attn_proj"] = blk.attn.proj
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2

        self.reset_parameter_standard()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return {"features": self.vit(x)}
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
    @property
    def feature_dim(self):
        return self.vit.num_features

    @torch.no_grad()
    def reset_parameters_svd(self):
        for mod in self.lora_modules.values():
            mod.reset_parameters_svd()

    def reset_parameter_standard(self):
        for mod in self.lora_modules.values():
            nn.init.normal_(mod.A, std=0.01)
            nn.init.zeros_(mod.B)

    @torch.no_grad()
    def merge_lora_weights(self, reset_after: bool = True):
        for mod in self.lora_modules.values():
            mod.merge_lora_weights()
        
        if reset_after:
            self.reset_parameter_standard()

    def return_lora_params(self):
        """返回所有 LoRA/DoRA 相关可训练参数"""
        params = []
        for n, p in self.lora_modules.named_parameters():
            if any(key in n for key in ["A", "B", "magnitude", "lora_scale"]):
                params.append(p)
        return params

    def return_lightweight_params(self):
        params = []
        for n, p in self.named_parameters():
            if "norm" in n or "cls_token" in n:
                params.append(p)
        return params

    def activate_lora_params(self, include_norms=True):
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.return_lora_params():
            p.requires_grad_(True)

        if include_norms:
            for p in self.return_lightweight_params():
                p.requires_grad_(True)


