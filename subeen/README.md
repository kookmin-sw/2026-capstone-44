# AerialVG에서 2D RoPE를 활용한 GroundingDINO Spatial Reasoning 개선

본 연구는 항공뷰 visual grounding **AerialVG**에서 GroundingDINO의 한계 — *decoder self-attention이 query 간 상대 위치 정보를 처리하지 못하는 문제* — 를 보고, **2D Rotary Position Embedding (RoPE-Mixed)** 을 도입하여 보완한다.

---

## 핵심 결과 (AerialVG Test Set, N=4,723)

| | Top-1 (%) | Top-5 (%) |
|---|---:|---:|
| **Baseline (zero-shot)** | 12.79 | 34.79 |
| **Ours (RoPE finetuned)** | **23.61** | **44.70** |
| Δ | **+10.82** | **+9.91** |

### Spatial-relation Split (가설 검증)

| Group | N | Baseline Top-1 | RoPE (Ours) Top-1 | Δ Top-1 | Baseline Top-5 | RoPE(Ours) Top-5 | Δ Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Spatial (binary relation)** | 1,048 | 10.21 | 22.04 | **+11.83** | 29.77 | 43.61 | **+13.84** |
| Non-spatial | 3,675 | 13.52 | 24.05 | +10.53 | 36.22 | 45.01 | +8.79 |

명시적 binary 공간관계(`left of`, `above`, `between` 등)가 포함된 caption에서 향상폭이 비대칭적으로 크다 (Top-5 +13.84%p vs +8.79%p, relative +46.5% vs +24.3%). 이는 RoPE의 spatial inductive bias가 효과의 근원임을 시사한다.

---

## 주요 Contribution

### 1. 문제점 진단
GroundingDINO의 6개 attention 위치 중 **decoder self-attention만이 유일하게 spatial inductive bias 없이 일반 `nn.MultiheadAttention`으로 처리**되는 query↔query 자리임을 확인. caption의 공간관계가 작동해야 하는 *유일한 자리*.

### 2. RoPE-Mixed
원본 `nn.MultiheadAttention` 한 덩어리를 `sa_q_proj / sa_k_proj / sa_v_proj / sa_out_proj` + `RopeMixed2D` 로 분해. reference_point의 (cx, cy)를 사용하여 Q, K에 학습 가능한 per-head 2D 회전을 적용 → query 간 attention score가 상대 위치에 의존하도록.

### 3. 정성/정량 양방향 검증
- 정량: Spatial vs Non-spatial 그룹별 향상폭 비대칭 확인
- 정성: flip-positive 상위 케이스에서 모두 binary spatial relation 단서를 모델이 정확히 활용 ([results/qualitative/](results/qualitative/))

---

## 디렉토리 구조

```
subeen/
├── README.md
├── GroundingDINO/
│   ├── train_change.py                    # 학습 스크립트 (RoPE finetuning)
│   ├── eval_aerial_paper.py               # baseline 평가
│   ├── eval_aerial_modified.py            # RoPE 평가 + spatial split 로직
│   └── groundingdino_modified/            # 수정/추가한 GDINO 파일
│       ├── rope_utils.py                  # RopeMixed2D 구현
│       ├── transformer.py                 # decoder self-attn 분해 + RoPE 삽입
│       ├── groundingdino.py               # RoPE 모델 빌드
│       ├── transoferm_original.py         # 원본 transformer (baseline 평가용)
│       └── groundingdino_original.py      # 원본 groundingdino (baseline 평가용)
└── results/
    ├── qualitative/                       # 정성 figure 6장 (case_*.png) — flip-positive 케이스
    └── limitation/                        # 한계 사례 figure 6장 (limitation_*.png)
```

---

## 설정

### 1. 데이터 준비

- AerialVG dataset: HuggingFace [`IPEC-COMMUNITY/aerial_vg`](https://huggingface.co/datasets/IPEC-COMMUNITY/aerial_vg)
- 포맷: ODVG (`vg_train_odvg.jsonl`, `vg_val_odvg.jsonl`, `vg_test_odvg.jsonl`)

### 2. 학습

```bash
python train_change.py
```

학습 결과는 `weights/finetuned_rope_v8/epoch_XX.pth` 로 저장됨. 본 보고서는 **epoch_12** 사용 (15 epoch 학습, best on val).

### 3. 평가

```bash
# Baseline (zero-shot)
python eval_aerial_paper.py

# Ours (RoPE finetuned)
python eval_aerial_modified.py
```

---

## 방법 (RoPE-Mixed 핵심)

### Decoder Self-Attention 재구성

원본:
```python
# transformer.py (원본)
q = k = tgt + tgt_query_pos                       # additive position embedding
tgt2 = self.self_attn(q, k, tgt, attn_mask=mask)[0]
```

수정 후:
```python
# transformer.py:836-936 (Ours)
q = self.sa_q_proj(tgt).view(B, N, H, Dh).transpose(1,2)
k = self.sa_k_proj(tgt).view(B, N, H, Dh).transpose(1,2)
v = self.sa_v_proj(tgt)
positions = tgt_reference_points[:, :, 0, :2].clamp(0,1)   # 정규화된 (cx, cy)
q, k = self.sa_rope(q, k, positions)              # Q, K에 RoPE-Mixed 회전 (V는 그대로)
attn = (q @ k.transpose(-2,-1)) * self.sa_scale
...
tgt2 = self.sa_out_proj(out)                      # zero-init → 초기엔 identity
```

### RoPE-Mixed (`rope_utils.py`)

학습 가능한 per-head 주파수 $\theta^x_t, \theta^y_t$ 로 임의 방향 표현:

$$\text{angle}_{h,t} = \theta^x_{h,t} \cdot p_x + \theta^y_{h,t} \cdot p_y$$

- **Axial-init**: paper의 Axial-RoPE 주파수와 수학적으로 동등하게 시작
- **Zero-init `sa_out_proj`**: epoch 0에서 self-attention 출력 = 0 → pretrained 호환


## 참고 문헌

- Liu, S., Zeng, Z., Ren, T., et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. arXiv 2303.05499, 2023.
- Su, J., Lu, Y., Pan, S., et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv 2104.09864, 2021.
- Heo, B., Park, S., Han, D., et al. *Rotary Position Embedding for Vision Transformer*. arXiv 2403.13298, 2024.
- AerialVG dataset, HuggingFace `IPEC-COMMUNITY/aerial_vg`.

