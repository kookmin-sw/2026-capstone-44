"""
2D RoPE utilities for vision.

References:
  - Heo et al., "Rotary Position Embedding for Vision Transformer", arXiv 2403.13298
  - GitHub: https://github.com/naver-ai/rope-vit
"""
import torch
import torch.nn as nn


# ==========================================================
# Core rotation helper  (GPT-NeoX style)
# ==========================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    GPT-NeoX / naver-rope-vit style:
      split at d/2, negate second half → [-x2, x1]

    Must be paired with embeddings built as cat([freqs, freqs]):
      emb[i]      = θ_i * pos          (first half)
      emb[i+d/2]  = θ_i * pos          (second half, same angle!)

    Then  q' = q*cos + rotate_half(q)*sin  gives a proper 2-D rotation
    for the (i, i+d/2) pair using the same angle θ_i*pos.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


# ==========================================================
# 1-D RoPE  (kept for reference / testing)
# ==========================================================

def build_1d_sincos(
    freq_seq: torch.Tensor,
    dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    freq_seq : (seq_len,)  — integer or continuous positions
    dim      : embedding dimension (must be even)
    returns  : cos, sin  each of shape (seq_len, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=freq_seq.device).float() / half))
    freqs = torch.einsum("n,d->nd", freq_seq.float(), inv_freq)  # (seq_len, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (seq_len, d)
    return emb.cos(), emb.sin()


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, N, head_dim)"""
    _, _, seq_len, head_dim = q.shape
    cos, sin = build_1d_sincos(torch.arange(seq_len, device=q.device), head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ==========================================================
# 2-D RoPE — Axial  (paper §3.2, Eq. 12-13)
# ==========================================================

def build_2d_axial_sincos(
    x_pos: torch.Tensor,
    y_pos: torch.Tensor,
    head_dim: int,
    base: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Paper Eq. 12-13: Axial 2D RoPE with interleaved x/y layout.

    Layout of the first-half dimensions (index 0 … head_dim/2 - 1):
      [cos(θ_0·x), cos(θ_0·y), cos(θ_1·x), cos(θ_1·y), ...]
    Second half duplicates (GPT-NeoX cat convention).

    x_pos, y_pos : [B, N]  normalised ∈ [0, 1]
    head_dim     : must be divisible by 4
    base         : 100 for vision (paper Eq. 13), 10000 for language
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D Axial RoPE"
    n_freq = head_dim // 4          # paper: d_head/4 frequencies per axis
    device = x_pos.device

    # θ_t = base^(-t / n_freq)  (paper Eq. 13 with base=100)
    inv_freq = 1.0 / (base ** (torch.arange(0, n_freq, device=device).float() / n_freq))

    freqs_x = torch.einsum("bn,d->bnd", x_pos.float(), inv_freq)  # [B, N, n_freq]
    freqs_y = torch.einsum("bn,d->bnd", y_pos.float(), inv_freq)

    # Interleave x and y at each frequency level → [B, N, head_dim/2]
    # stack → [B, N, n_freq, 2], then flatten last two dims
    freqs_xy = torch.stack([freqs_x, freqs_y], dim=-1).flatten(-2)  # [B, N, 2*n_freq]

    # GPT-NeoX cat convention → [B, N, head_dim]
    emb = torch.cat([freqs_xy, freqs_xy], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope_2d_axial(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    base: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k      : [B, H, N, head_dim]
    positions : [B, N, 2]  (cx, cy) ∈ [0, 1]
    """
    B, H, N, head_dim = q.shape
    cos, sin = build_2d_axial_sincos(
        positions[..., 0], positions[..., 1], head_dim, base
    )                                                   # [B, N, head_dim]
    cos = cos.unsqueeze(1).expand(-1, H, -1, -1)        # [B, H, N, head_dim]
    sin = sin.unsqueeze(1).expand(-1, H, -1, -1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ==========================================================
# 2-D RoPE — Mixed with learnable frequencies  (paper §3.2, Eq. 14)
# ==========================================================

class RopeMixed2D(nn.Module):
    """
    Paper Eq. 14: RoPE-Mixed.

    R(n, t) = e^{i · (θˣ_t · pˣ + θʸ_t · pʸ)}

    (θˣ_t, θʸ_t) are learnable per-head parameters initialised from
    the Axial frequencies so training starts from a sensible prior.

    Paper says this is superior to Axial because it can capture
    diagonal directions (neither purely horizontal nor vertical).
    Only ~d learnable params per attention layer — negligible overhead.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        base: float = 100.0,
        init: str = "axial",
    ):
        super().__init__()
        assert head_dim % 2 == 0
        n_freq = head_dim // 2          # d_head/2 frequencies (paper §3.2)
        n_axial = n_freq // 2           # d_head/4 axial frequencies per axis

        # init="axial": epoch 0에서 Axial-RoPE(paper Eq. 12-13)와 수학적으로 동등.
        # sa_out_proj zero-init이 self-attn residual을 0으로 막아 pretrained 동작은
        # 보호되면서, sa_out_proj가 풀리는 순간 axial spatial prior가 즉시 작동.
        # 학습이 진행되면 theta가 axial에서 diagonal 방향으로 풀려나감.
        if init == "zero":
            tx = torch.zeros(num_heads, n_freq)
            ty = torch.zeros(num_heads, n_freq)
        elif init == "axial":
            # axial layout (build_2d_axial_sincos 참조):
            #   freqs = [θ_0·x, θ_0·y, θ_1·x, θ_1·y, ...]  (interleaved)
            # 짝수 인덱스(0,2,4,...)에 x축 주파수, 홀수 인덱스(1,3,5,...)에 y축 주파수.
            inv_freq = 1.0 / (base ** (torch.arange(n_axial).float() / n_axial))
            tx_1d = torch.zeros(n_freq)
            ty_1d = torch.zeros(n_freq)
            tx_1d[0::2] = inv_freq
            ty_1d[1::2] = inv_freq
            tx = tx_1d.unsqueeze(0).expand(num_heads, -1).clone()
            ty = ty_1d.unsqueeze(0).expand(num_heads, -1).clone()
        else:
            raise ValueError(f"unknown init mode: {init!r} (expected 'axial' or 'zero')")

        self.theta_x = nn.Parameter(tx)  # [H, n_freq]
        self.theta_y = nn.Parameter(ty)  # [H, n_freq]

    def _get_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions : [B, N, 2]
        returns cos, sin : [B, H, N, head_dim]
        """
        px = positions[..., 0]   # [B, N]
        py = positions[..., 1]

        # angle_t = θˣ_t · pˣ + θʸ_t · pʸ  for each head and freq
        # einsum: "bn, hd -> bhnd"
        angles = (torch.einsum("bn,hd->bhnd", px, self.theta_x)
                + torch.einsum("bn,hd->bhnd", py, self.theta_y))  # [B, H, N, n_freq]

        # GPT-NeoX cat convention → [B, H, N, head_dim]
        emb = torch.cat([angles, angles], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k      : [B, H, N, head_dim]
        positions : [B, N, 2]  (cx, cy) ∈ [0, 1]
        """
        cos, sin = self._get_cos_sin(positions)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ==========================================================
# 1-D RoPE — Learnable (for text cross-attention)
# ==========================================================

class RopeSeq1D(nn.Module):
    """
    Learnable 1D RoPE for sequential positions (text tokens, visual query indices).

    Q (visual queries) and K (text tokens) each get their own positional rotation
    based on normalized sequential positions in [0, 1].
    Zero-init: no rotation at epoch 0 → pretrained ca_text behaviour preserved.
    """

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        assert head_dim % 2 == 0
        n_freq = head_dim // 2
        self.theta_q = nn.Parameter(torch.zeros(num_heads, n_freq))  # [H, n_freq]
        self.theta_k = nn.Parameter(torch.zeros(num_heads, n_freq))  # [H, n_freq]

    def _get_cos_sin(self, pos: torch.Tensor, theta: torch.Tensor):
        """
        pos   : [B, N]  normalized positions ∈ [0, 1]
        theta : [H, n_freq]
        returns cos, sin : [B, H, N, head_dim]
        """
        angles = torch.einsum("bn,hd->bhnd", pos, theta)       # [B, H, N, n_freq]
        emb = torch.cat([angles, angles], dim=-1)               # [B, H, N, head_dim]
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: torch.Tensor,          # [B, H, Nq, head_dim]
        k: torch.Tensor,          # [B, H, Lk, head_dim]
        q_pos: torch.Tensor,      # [B, Nq]  normalized ∈ [0, 1]
        k_pos: torch.Tensor,      # [B, Lk]  normalized ∈ [0, 1]
    ):
        cos_q, sin_q = self._get_cos_sin(q_pos, self.theta_q)
        cos_k, sin_k = self._get_cos_sin(k_pos, self.theta_k)
        return (q * cos_q + rotate_half(q) * sin_q,
                k * cos_k + rotate_half(k) * sin_k)
