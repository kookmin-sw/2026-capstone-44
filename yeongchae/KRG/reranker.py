import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AerialReranker(nn.Module):
    def __init__(
        self,
        logit_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        knn_k: int = 5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.knn_k = knn_k

        input_dim = logit_dim + 4 + 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        edge_feat_dim = 8
        self.edge_proj = nn.Linear(edge_feat_dim, nhead)

        self.attn_layers = nn.ModuleList([
            _RelationAwareAttentionLayer(hidden_dim, nhead)
            for _ in range(num_layers)
        ])

        self.score_head = nn.Linear(hidden_dim, 1)

    def _build_knn_mask(self, boxes):
        B, K, _ = boxes.shape
        cx = boxes[:, :, 0]
        cy = boxes[:, :, 1]

        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        eye = torch.eye(K, device=boxes.device, dtype=torch.bool).unsqueeze(0)
        dist_for_knn = dist.masked_fill(eye, float("inf"))

        k = min(self.knn_k, K - 1)
        _, knn_idx = torch.topk(dist_for_knn, k=k, dim=2, largest=False)

        mask = eye.expand(B, K, K).clone()
        mask.scatter_(2, knn_idx, True)
        mask = mask | mask.transpose(1, 2)

        return mask

    def _build_edge_features(self, boxes):
        cx = boxes[:, :, 0]
        cy = boxes[:, :, 1]
        w  = boxes[:, :, 2]
        h  = boxes[:, :, 3]

        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        log_w_ratio = torch.log(w.unsqueeze(2) / (w.unsqueeze(1) + 1e-8) + 1e-8)
        log_h_ratio = torch.log(h.unsqueeze(2) / (h.unsqueeze(1) + 1e-8) + 1e-8)

        area_i = (w * h).unsqueeze(2)
        area_j = (w * h).unsqueeze(1)
        area_ratio = area_i / (area_j + 1e-8)

        left_flag  = (cx.unsqueeze(2) < cx.unsqueeze(1)).float()
        above_flag = (cy.unsqueeze(2) < cy.unsqueeze(1)).float()

        edge_feat = torch.stack(
            [dx, dy, dist, log_w_ratio, log_h_ratio,
             area_ratio, left_flag, above_flag],
            dim=-1,
        )

        return edge_feat

    def forward(self, topk_logits, topk_boxes, topk_scores):
        B, K, _ = topk_logits.shape

        candidate_input = torch.cat(
            [topk_logits, topk_boxes, topk_scores.unsqueeze(-1)],
            dim=-1,
        )
        z = self.input_proj(candidate_input)

        knn_mask = self._build_knn_mask(topk_boxes)

        edge_feat = self._build_edge_features(topk_boxes)
        attn_bias = self.edge_proj(edge_feat)
        attn_bias = attn_bias.permute(0, 3, 1, 2)

        attn_mask = ~knn_mask
        attn_mask = attn_mask.unsqueeze(1).expand_as(attn_bias)
        attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))

        for layer in self.attn_layers:
            z = layer(z, attn_bias)

        rerank_score = self.score_head(z).squeeze(-1)

        return rerank_score


class _RelationAwareAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, nhead):
        super().__init__()
        assert hidden_dim % nhead == 0
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, attn_bias):
        B, K, C = x.shape
        H, D = self.nhead, self.head_dim

        Q  = self.q_proj(x).reshape(B, K, H, D).permute(0, 2, 1, 3)
        Kk = self.k_proj(x).reshape(B, K, H, D).permute(0, 2, 1, 3)
        V  = self.v_proj(x).reshape(B, K, H, D).permute(0, 2, 1, 3)

        attn_score = torch.matmul(Q, Kk.transpose(-2, -1)) / self.scale
        attn_score = attn_score + attn_bias

        attn_weight = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weight, V)
        out = out.permute(0, 2, 1, 3).reshape(B, K, C)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))

        return x
