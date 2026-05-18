"""
Spatial Relation Bias Module (SRBM)

1. Geometric feature (sinθ, cosθ, dist, dw, dh) — no params, pure math
2. MLP 5→64→1: scalar bias per pair  (~400 params)
3. Self-attention: softmax(QK^T/√d + bias_ij)
4. Cross-attention with text
5. FFN → ContrastiveEmbed re-scoring
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEmbed(nn.Module):
    def __init__(self, max_text_len: int = 256):
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x: torch.Tensor, text_dict: dict) -> torch.Tensor:
        enc  = text_dict["encoded_text"]     # [B, L, d]
        mask = text_dict["text_token_mask"]  # [B, L]
        logits = torch.einsum("bnd,bsd->bns", x, enc)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        seq = logits.size(-1)
        if seq < self.max_text_len:
            pad = torch.full(
                (logits.size(0), logits.size(1), self.max_text_len - seq),
                float("-inf"), device=logits.device, dtype=logits.dtype,
            )
            logits = torch.cat([logits, pad], dim=-1)
        else:
            logits = logits[:, :, :self.max_text_len]
        return logits


class SpatialBiasMLP(nn.Module):
    """5-dim geo feature → scalar attention bias.  ~400 params."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : [B, K, 4]  cxcywh normalized
        returns: [B, K, K]  scalar bias per pair
        """
        B, K, _ = coords.shape
        cx, cy, w, h = coords.unbind(-1)

        dx   = cx.unsqueeze(2) - cx.unsqueeze(1)          # [B, K, K]
        dy   = cy.unsqueeze(2) - cy.unsqueeze(1)
        dist = (dx.pow(2) + dy.pow(2)).sqrt().clamp(min=1e-6)
        sinθ = dy / dist
        cosθ = dx / dist
        dw   = (w.unsqueeze(2) / w.unsqueeze(1).clamp(min=1e-6)).log().clamp(-5, 5)
        dh   = (h.unsqueeze(2) / h.unsqueeze(1).clamp(min=1e-6)).log().clamp(-5, 5)

        geo = torch.stack([sinθ, cosθ, dist, dw, dh], dim=-1)  # [B, K, K, 5]
        return self.mlp(geo).squeeze(-1)                         # [B, K, K]


class SpatialBiasSelfAttention(nn.Module):
    """Multi-head self-attention with spatial scalar bias on logits."""
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, K, d]
        bias: [B, K, K]  spatial scalar bias
        """
        B, K, d = x.shape
        H, Hd   = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, K, H, Hd).permute(0, 2, 1, 3)  # [B,H,K,Hd]
        Kp = self.k_proj(x).view(B, K, H, Hd).permute(0, 2, 1, 3)
        V  = self.v_proj(x).view(B, K, H, Hd).permute(0, 2, 1, 3)

        attn = torch.einsum("bhid,bhjd->bhij", Q, Kp) * self.scale  # [B,H,K,K]
        attn = attn + bias.unsqueeze(1)                              # broadcast across heads
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, K, d)
        return self.norm(x + self.out_proj(out))


class SpatialRelationBiasModule(nn.Module):
    """
    SRBM: Spatial Relation Bias Module

    Params breakdown:
      SpatialBiasMLP  : ~400
      SpatialBiasSelfAttention × num_layers : ~800K
      CrossAttention  × num_layers          : ~800K
      FFN             × num_layers          : ~500K
      ContrastiveEmbed: 0
    """
    def __init__(
        self,
        d_model:      int = 256,
        num_heads:    int = 8,
        num_layers:   int = 3,
        topk:         int = 15,
        max_text_len: int = 256,
        ffn_dim:      int = 1024,
    ):
        super().__init__()
        self.topk         = topk
        self.max_text_len = max_text_len

        # Shared spatial bias MLP (used in every layer)
        self.spatial_bias = SpatialBiasMLP()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn":  SpatialBiasSelfAttention(d_model, num_heads),
                "cross_attn": nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, d_model),
                ),
                "ffn_norm": nn.LayerNorm(d_model),
            }))

        self.fc_logits = ContrastiveEmbed(max_text_len)

    def forward(
        self,
        features:  torch.Tensor,  # [B, K, 256]
        coords:    torch.Tensor,  # [B, K, 4]  cxcywh
        text_dict: dict,
    ) -> torch.Tensor:            # [B, K, max_text_len]

        text_feat = text_dict["encoded_text"]     # [B, L, 256]
        text_mask = text_dict["text_token_mask"]  # [B, L]

        # Compute spatial bias once — shared across all layers
        bias = self.spatial_bias(coords)           # [B, K, K]

        x = features
        for layer in self.layers:
            # 1. Spatial bias self-attention
            x = layer["self_attn"](x, bias)

            # 2. Cross-attention with text
            ca, _ = layer["cross_attn"](
                x, text_feat, text_feat,
                key_padding_mask=~text_mask,
            )
            x = layer["cross_norm"](x + ca)

            # 3. FFN
            x = layer["ffn_norm"](x + layer["ffn"](x))

        return self.fc_logits(x, text_dict)        # [B, K, max_text_len]
