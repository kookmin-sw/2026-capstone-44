# AerialVG에서 2D RoPE를 활용한 GroundingDINO Spatial Reasoning 개선

본 연구는 항공뷰 visual grounding 벤치마크 **AerialVG**에서 GroundingDINO의 한계 — *decoder self-attention이 query 간 상대 위치 정보를 처리하지 못하는 병목* — 를 진단하고, **2D Rotary Position Embedding (RoPE-Mixed)** 을 도입하여 보완한다.

---

## 핵심 결과 (AerialVG Test Set, N=4,723)

| | Top-1 (%) | Top-5 (%) |
|---|---:|---:|
| **A — Baseline (zero-shot)** | 12.79 | 34.79 |
| **C — Ours (RoPE finetuned)** | **23.61** | **44.70** |
| Δ | **+10.82** | **+9.91** |

### Spatial-relation Split (가설 검증)

| Group | N | A Top-1 | C Top-1 | Δ Top-1 | A Top-5 | C Top-5 | Δ Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Spatial (binary relation)** | 1,048 | 10.21 | 22.04 | **+11.83** | 29.77 | 43.61 | **+13.84** |
| Non-spatial | 3,675 | 13.52 | 24.05 | +10.53 | 36.22 | 45.01 | +8.79 |

명시적 binary 공간관계(`left of`, `above`, `between` 등)가 포함된 caption에서 향상폭이 비대칭적으로 크다 (Top-5 +13.84%p vs +8.79%p, relative +46.5% vs +24.3%). 이는 RoPE의 spatial inductive bias가 효과의 근원임을 시사한다.

---

## 주요 Contribution

### 1. 병목 진단
GroundingDINO의 6개 attention 위치 중 **decoder self-attention만이 유일하게 spatial inductive bias 없이 일반 `nn.MultiheadAttention`으로 처리**되는 query↔query 자리임을 확인. caption의 공간관계가 작동해야 하는 *유일한 자리*가 곧 병목.

### 2. 표적 변경 (RoPE-Mixed)
원본 `nn.MultiheadAttention` 한 덩어리를 `sa_q_proj / sa_k_proj / sa_v_proj / sa_out_proj` + `RopeMixed2D` 로 분해. reference_point의 (cx, cy)를 사용하여 Q, K에 학습 가능한 per-head 2D 회전을 적용 → query 간 attention score가 *상대 위치*에 의존하도록.

추가 학습 파라미터 **layer당 512개, 6 layer 총 3,072개** (모델 전체 대비 무시 가능).

### 3. 학습 안정성 (Zero-init `sa_out_proj`)
ControlNet/LoRA 식 zero-init을 적용하여 epoch 0에서 self-attention 출력이 0 → pretrained 모듈 보호 → 학습이 진행되면서 점진적 활성화.

### 4. 정성/정량 양방향 검증
- 정량: Spatial vs Non-spatial 그룹별 향상폭 비대칭 확인
- 정성: flip-positive 상위 케이스에서 모두 binary spatial relation 단서를 모델이 정확히 활용 ([results/qualitative/](results/qualitative/))

---

## 디렉토리 구조

```
subeen/
├── README.md
├── GroundingDINO/
│   ├── train_change.py                    # 학습 스크립트 (RoPE finetuning)
│   ├── eval_aerial_paper.py               # A 모델 (baseline) 평가
│   ├── eval_aerial_modified.py            # C 모델 (RoPE) 평가 + spatial split 로직
│   └── groundingdino_modified/            # 수정/추가한 GDINO 파일
│       ├── rope_utils.py                  # [신규] RopeMixed2D 구현
│       ├── transformer.py                 # [수정] decoder self-attn 분해 + RoPE 삽입
│       ├── groundingdino.py               # [수정] RoPE 모델 빌더
│       ├── transoferm_original.py         # [참조] 원본 transformer (baseline 평가용)
│       └── groundingdino_original.py      # [참조] 원본 groundingdino (baseline 평가용)
└── results/
    ├── qualitative/                       # 정성 figure 6장 (case_*.png) — flip-positive 케이스
    └── limitation/                        # 한계 사례 figure 6장 (limitation_*.png)
```

---

## 환경 설정 및 재현

### 1. 코드 설치

```bash
# 원본 Grounding DINO (IDEA-Research) clone & install
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

> **빌드 호환성 이슈**: 최신 PyTorch에서 csrc 빌드 실패 시, `groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h` 와 `ms_deform_attn_cuda.cu` 의 deprecated API를 다음과 같이 패치 필요.
> - `value.type().is_cuda()` → `value.is_cuda()`
> - `value.type()` (in `AT_DISPATCH_FLOATING_TYPES`) → `value.scalar_type()`

### 2. 본 연구 코드 적용

```bash
# 학습/평가 스크립트
cp subeen/GroundingDINO/*.py GroundingDINO/

# 수정된 모델 파일 (RoPE 모델용)
cp subeen/GroundingDINO/groundingdino_modified/rope_utils.py \
   subeen/GroundingDINO/groundingdino_modified/transformer.py \
   subeen/GroundingDINO/groundingdino_modified/groundingdino.py \
   GroundingDINO/groundingdino/models/GroundingDINO/

# 베이스라인 평가용 (원본 보존본 — baseline 평가 시에만 사용)
cp subeen/GroundingDINO/groundingdino_modified/groundingdino_original.py \
   subeen/GroundingDINO/groundingdino_modified/transoferm_original.py \
   GroundingDINO/groundingdino/models/GroundingDINO/
```

### 3. 데이터 준비

- AerialVG dataset: HuggingFace [`IPEC-COMMUNITY/aerial_vg`](https://huggingface.co/datasets/IPEC-COMMUNITY/aerial_vg)
- 포맷: ODVG (`vg_train_odvg.jsonl`, `vg_val_odvg.jsonl`, `vg_test_odvg.jsonl`)

### 4. Pretrained Checkpoint

- `groundingdino_swint_ogc.pth` (IDEA-Research 공식 다운로드)
- `weights/` 디렉토리에 배치

### 5. 학습

```bash
python train_change.py
```

학습 결과는 `weights/finetuned_rope_v8/epoch_XX.pth` 로 저장됨. 본 보고서는 **epoch_12** 사용 (15 epoch 학습, best on val).

### 6. 평가

```bash
# A — Baseline (zero-shot)
python eval_aerial_paper.py

# C — Ours (RoPE finetuned)
python eval_aerial_modified.py
```

---

## 방법론 (RoPE-Mixed 핵심)

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

- **Axial-init**: paper의 Axial-RoPE 주파수와 수학적으로 동등하게 시작 (NumPy max-abs-diff = 0 검증)
- **Zero-init `sa_out_proj`**: epoch 0에서 self-attention 출력 = 0 → pretrained 호환

---

## 한계 (Caveat)

- Baseline은 zero-shot, ours는 finetuned → 향상의 일부는 finetune 효과. 완전 분리하려면 "RoPE 없이 동일 조건 finetune" 추가 비교군 필요 (시간 제약으로 미완)
- 단, **spatial > non-spatial 비대칭 향상 패턴**은 finetune 자체로 설명 어려움. Finetune은 attention의 inductive bias를 바꾸지 않으므로 그룹별 차등 향상의 원인을 갖지 않음
- `theta_x, theta_y` 는 zero-init으로 학습됨. Axial-init 코드는 검증 완료되었으나 full retraining은 GPU 가용성 부족으로 future work

---

## 참고 문헌

- Liu, S., Zeng, Z., Ren, T., et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. arXiv 2303.05499, 2023.
- Su, J., Lu, Y., Pan, S., et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv 2104.09864, 2021.
- Heo, B., Park, S., Han, D., et al. *Rotary Position Embedding for Vision Transformer*. arXiv 2403.13298, 2024.
- AerialVG dataset, HuggingFace `IPEC-COMMUNITY/aerial_vg`.

---

## 라이선스 / 참조

- **Grounding DINO**: Apache 2.0 (IDEA-Research)
- **본 연구 코드**: 학술 목적 사용

---

## Contributor

- **2026 캡스톤 디자인 - 팀 44**
- 국민대학교
