import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

class CrossSelfRelationTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8,num_layers=3):
        super(CrossSelfRelationTransformer, self).__init__()

        # 自注意力降维
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=num_heads, batch_first=True)
        self.linear_proj = nn.Linear(d_model * 2, d_model)  # 新增线性层降维
        self.self_attn_norm = nn.LayerNorm(d_model)

        # 交替堆叠 Cross-Attention 和 Self-Attention 层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(d_model,eps=1e-6),
                "self_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                "self_norm": nn.LayerNorm(d_model,eps=1e-6)
            })
            for _ in range(num_layers)
        ])

        # 映射到 logits 维度
        self.fc_logits = ContrastiveEmbed()
        for layer in self.layers:
            nn.init.xavier_uniform_(layer["cross_attn"].in_proj_weight)
            nn.init.xavier_uniform_(layer["self_attn"].in_proj_weight)
            nn.init.xavier_uniform_(layer["cross_attn"].out_proj.weight)
            nn.init.xavier_uniform_(layer["self_attn"].out_proj.weight)

    def forward(self, image_features, text_dict):
        text_features = text_dict['encoded_text']
        text_attention_mask = text_dict["text_token_mask"]
        # print('image_features',image_features)
        bs, num, d_model = image_features.shape
        
        # Step 1: 两两组合图像特征
        combined_features = torch.cat([
            image_features.unsqueeze(2).expand(-1, -1, num, -1),
            image_features.unsqueeze(1).expand(-1, num, -1, -1)
        ], dim=-1)  # [bs, num, num, d_model * 2]
        combined_features = combined_features.view(bs, num * num, d_model * 2)  # [bs, num^2, d_model * 2]

        # 使用自注意力直接降维
        relation_features, _ = self.self_attention(combined_features, combined_features, combined_features)
        relation_features = self.linear_proj(relation_features)  # 降维至 [bs, num^2, d_model]
        relation_features = self.self_attn_norm(relation_features)
        # print('relation_features',relation_features.shape,relation_features)
        # print('text_attention_mask',text_attention_mask.shape,text_attention_mask)

        # Step 2: 交替堆叠 Cross-Attention 和 Self-Attention
        for layer in self.layers:
            # Cross-Attention
            cross_attn_output, _ = layer["cross_attn"](
                query=relation_features,          # 使用图像关系特征作为 query
                key=text_features,                # 使用文本特征作为 key
                value=text_features,              # 使用文本特征作为 value
                key_padding_mask= ~text_attention_mask  # 使用文本的掩码
            )
            relation_features = layer["cross_norm"](cross_attn_output + relation_features)  # 残差连接 + LayerNorm
            relation_features = F.relu(relation_features)

            # Self-Attention
            self_attn_output, _ = layer["self_attn"](relation_features, relation_features, relation_features)
            relation_features = layer["self_norm"](self_attn_output + relation_features)  # 残差连接 + LayerNorm

        # Step 3: 映射到 logits 维度
        logits = self.fc_logits(relation_features,text_dict)
        # print('logits',logits.shape,logits)
        logits_matrix = logits.view(bs, num, num,d_model)


        # 置零对角线元素
        # mask = 1 - torch.eye(num, device=logits.device).unsqueeze(0).unsqueeze(-1)  # [1, num, num, 1]
        # logits_matrix = logits_matrix * mask
        # logits_matrix = logits_matrix * (1 - torch.eye(num, device=logits.device).unsqueeze(0))

        # 分别在两个 num 维度上取最大值，得到两个 [bs, num] 的分数
        max_score_dim1= logits_matrix.mean(dim=1)  # 沿第一个 num 维度取最大值
        max_score_dim2 = logits_matrix.mean(dim=2)  # 沿第二个 num 维度取最大值

        # 取平均，得到每个图像特征的最终分数
        final_scores = (max_score_dim1 + max_score_dim2) / 2  # [bs, num]
        return final_scores

# # 测试
# batch_size = 4
# num_relations = 15
# d_model = 256
# num_tokens = 20
# text_dim = 256
# logits_dim = 256

# # 假设输入
# relation_features = torch.rand(batch_size, num_relations, d_model)
# text_features = torch.rand(batch_size, num_tokens, text_dim)
# text_attention_mask = torch.ones(batch_size, num_tokens).bool()

# model = CrossSelfRelationTransformer(d_model=d_model, num_heads=8, num_relations=num_relations, text_dim=text_dim, logits_dim=logits_dim)
# logits = model(relation_features, text_features)

# print("Logits shape:", logits.shape)  # 输出形状应为 (batch_size, num_relations, logits_dim)

def build_relation_transformer(args):
    """
    构建 CrossSelfRelationTransformer 模型。

    参数:
    - args (Namespace): 包含所有模型超参数的命名空间对象。

    返回:
    - CrossSelfRelationTransformer 模型实例
    """
    return CrossSelfRelationTransformer(
        d_model=args.hidden_dim,
        num_heads=args.nheads,
        num_layers=args.relation_num_layers
    )


class RoleGatedEvidenceModule(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.0, role_gate_mode="learned", aux_topk=4):
        super().__init__()
        if role_gate_mode not in {"learned", "none", "shuffle"}:
            raise ValueError(f"Unknown role_gate_mode: {role_gate_mode}")
        self.role_gate_mode = role_gate_mode
        self.aux_topk = max(1, int(aux_topk))
        self.role_text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.role_context_norm = nn.LayerNorm(d_model)
        self.role_mlp = MLP(d_model * 2, d_model, 3, 3)
        self.gt_text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gt_aux_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.evidence_norm = nn.LayerNorm(d_model)
        self.evidence_ffn = MLP(d_model, d_model, d_model, 2)
        self.evidence_out_norm = nn.LayerNorm(d_model)
        self.refined_score_head = ContrastiveEmbed()

    @staticmethod
    def _cyclic_shift_values(values, selected_valid_mask):
        shifted_values = values.clone()
        for batch_idx in range(values.shape[0]):
            valid_indices = selected_valid_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_indices.numel() <= 1:
                continue
            shifted_values[batch_idx, valid_indices] = values[batch_idx, valid_indices].roll(shifts=1, dims=0)
        return shifted_values

    @staticmethod
    def _first_valid_indices(selected_valid_mask):
        return selected_valid_mask.to(dtype=torch.long).argmax(dim=1)

    @staticmethod
    def _masked_argmax(scores, valid_mask):
        return scores.masked_fill(~valid_mask, float("-inf")).argmax(dim=1)

    @staticmethod
    def _gather_features(features, indices):
        return torch.gather(
            features,
            1,
            indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
        )

    def _select_gt_and_aux(self, role_probs, selected_valid_mask):
        batch_size, num_selected, _ = role_probs.shape
        candidate_indices = torch.arange(num_selected, device=role_probs.device).unsqueeze(0).expand(batch_size, -1)

        if self.role_gate_mode == "none":
            gt_indices = self._first_valid_indices(selected_valid_mask)
            aux_indices = candidate_indices
            aux_valid_mask = selected_valid_mask & candidate_indices.ne(gt_indices.unsqueeze(1))
            return gt_indices, aux_indices, aux_valid_mask

        selection_probs = role_probs
        if self.role_gate_mode == "shuffle":
            selection_probs = self._cyclic_shift_values(role_probs, selected_valid_mask)
        elif self.role_gate_mode != "learned":
            raise ValueError(f"Unknown role_gate_mode: {self.role_gate_mode}")

        gt_indices = self._masked_argmax(selection_probs[..., 0], selected_valid_mask)
        aux_valid_candidates = selected_valid_mask & candidate_indices.ne(gt_indices.unsqueeze(1))
        aux_scores = selection_probs[..., 1].masked_fill(~aux_valid_candidates, float("-inf"))
        aux_topk = min(self.aux_topk, num_selected)
        aux_values, aux_indices = torch.topk(aux_scores, aux_topk, dim=1)
        aux_valid_mask = torch.isfinite(aux_values)
        return gt_indices, aux_indices, aux_valid_mask

    def _build_role_masks(self, role_probs, selected_valid_mask):
        gt_indices, aux_indices, aux_valid_mask = self._select_gt_and_aux(role_probs, selected_valid_mask)
        batch_size, num_selected, _ = role_probs.shape
        gt_mask = torch.zeros((batch_size, num_selected), dtype=torch.bool, device=role_probs.device)
        gt_mask.scatter_(1, gt_indices.unsqueeze(1), True)
        gt_mask = gt_mask & selected_valid_mask

        aux_gate = torch.zeros((batch_size, num_selected, 1), dtype=role_probs.dtype, device=role_probs.device)
        aux_gate.scatter_(1, aux_indices.unsqueeze(-1), aux_valid_mask.unsqueeze(-1).to(role_probs.dtype))
        aux_gate = aux_gate * selected_valid_mask.unsqueeze(-1).to(role_probs.dtype)
        return gt_indices, gt_mask, aux_indices, aux_valid_mask, aux_gate

    def _scatter_attention_weights(self, gt_indices, aux_indices, aux_valid_mask, aux_attn_weights, num_selected):
        batch_size = gt_indices.shape[0]
        full_weights = aux_attn_weights.new_full((batch_size, num_selected, num_selected), -20.0)
        for batch_idx in range(batch_size):
            valid_aux = aux_valid_mask[batch_idx]
            if not valid_aux.any():
                continue
            row = gt_indices[batch_idx]
            cols = aux_indices[batch_idx][valid_aux]
            full_weights[batch_idx, row, cols] = aux_attn_weights[batch_idx, 0, valid_aux]
        return full_weights

    def forward(self, selected_features, text_dict, selected_valid_mask=None):
        text_features = text_dict["encoded_text"]
        text_attention_mask = text_dict["text_token_mask"]
        if selected_valid_mask is None:
            selected_valid_mask = torch.ones(
                selected_features.shape[:2],
                dtype=torch.bool,
                device=selected_features.device,
            )
        else:
            selected_valid_mask = selected_valid_mask.to(device=selected_features.device, dtype=torch.bool)

        role_text_context, _ = self.role_text_attn(
            query=selected_features,
            key=text_features,
            value=text_features,
            key_padding_mask=~text_attention_mask,
        )
        role_text_context = self.role_context_norm(role_text_context + selected_features)
        role_inputs = torch.cat([selected_features, role_text_context], dim=-1)
        role_logits = self.role_mlp(role_inputs)
        invalid_template = role_logits.new_tensor([0.0, 0.0, 1.0]).view(1, 1, 3)
        role_logits = torch.where(selected_valid_mask.unsqueeze(-1), role_logits, invalid_template)
        role_probs = role_logits.softmax(dim=-1)

        gt_indices, gt_mask, aux_indices, aux_valid_mask, aux_gate = self._build_role_masks(
            role_probs,
            selected_valid_mask,
        )
        gt_features = self._gather_features(selected_features, gt_indices.unsqueeze(1))
        aux_features = self._gather_features(selected_features, aux_indices)
        aux_features = aux_features * aux_valid_mask.unsqueeze(-1).to(dtype=aux_features.dtype)
        safe_aux_valid_mask = aux_valid_mask.clone()
        empty_aux_mask = ~safe_aux_valid_mask.any(dim=1)
        if empty_aux_mask.any():
            safe_aux_valid_mask[empty_aux_mask, 0] = True

        text_conditioned_gt, _ = self.gt_text_attn(
            query=gt_features,
            key=text_features,
            value=text_features,
            key_padding_mask=~text_attention_mask,
        )

        evidence_context, aux_attn_weights = self.gt_aux_attn(
            query=text_conditioned_gt,
            key=aux_features,
            value=aux_features,
            key_padding_mask=~safe_aux_valid_mask,
        )
        refined_gt_features = self.evidence_norm(gt_features + evidence_context)
        refined_gt_features = self.evidence_out_norm(
            refined_gt_features + self.evidence_ffn(refined_gt_features)
        )
        refined_features = selected_features.clone()
        refined_features = refined_features.scatter(
            1,
            gt_indices.view(-1, 1, 1).expand(-1, -1, selected_features.shape[-1]),
            refined_gt_features,
        )
        refined_features = refined_features * selected_valid_mask.unsqueeze(-1)
        refined_logits = self.refined_score_head(refined_features, text_dict)
        full_aux_attn_weights = self._scatter_attention_weights(
            gt_indices,
            aux_indices,
            aux_valid_mask,
            aux_attn_weights,
            selected_features.shape[1],
        )

        return {
            "role_logits": role_logits,
            "role_probs": role_probs,
            "role_gate_mode": self.role_gate_mode,
            "aux_gate": aux_gate,
            "gt_indices": gt_indices,
            "gt_mask": gt_mask,
            "aux_indices": aux_indices,
            "aux_valid_mask": aux_valid_mask,
            "role_text_context": role_text_context,
            "text_conditioned_gt": text_conditioned_gt,
            "aux_attention_weights": full_aux_attn_weights,
            "refined_features": refined_features,
            "refined_valid_mask": gt_mask,
            "refined_logits": refined_logits,
        }


def build_role_evidence_module(args):
    return RoleGatedEvidenceModule(
        d_model=args.hidden_dim,
        num_heads=args.nheads,
        dropout=getattr(args, "dropout", 0.0),
        role_gate_mode=getattr(args, "role_gate_mode", "learned"),
        aux_topk=getattr(args, "method2_aux_topk", 4),
    )
