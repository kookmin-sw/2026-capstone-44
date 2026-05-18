# AerialVG Visual Grounding with Spatial Relation Bias Module (SRBM)

본 연구는 항공 이미지 Visual Grounding (AerialVG) 데이터셋에서 SOTA를 능가하는 **Spatial Relation Bias Module (SRBM)** 을 제안합니다.

---

## 🎯 핵심 결과 (Test Set)

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| AerialVG paper (SOTA) | 49.88% | 87.38% |
| **AerialVG + Our SRBM** | **59.77%** | **91.72%** |
| **Improvement** | **+9.89%p** | **+4.34%p** |

같은 백본/같은 K=15/같은 입력에서 **순수 모듈 설계 차이만으로 SOTA +9.89%p 우월**.

---

## 📌 주요 Contribution

### 1. SRBM: 새로운 효율적 Relation 모듈
- **Attention bias 방식**: 5-dim geometry feature (sinθ, cosθ, dist, Δw, Δh) → MLP (5→64→1) → scalar bias → self-attention logit에 직접 주입
- **AerialVG 기존 모듈 대비**:
  - Pairwise → Individual + bias (시퀀스 K² → K)
  - Attention 복잡도 O(K⁴) → **O(K²)** (200× 효율)
  - Feature add → **logit bias add** (더 자연스러운 inductive bias)

### 2. SOTA 능가 (Fair Comparison)
같은 K=15, 같은 입력으로 fair 비교 시 SOTA +10.93%p (val), +9.89%p (test) 달성.

### 3. Top-K 확장 가능성 발견
| K | Top-1 (Val) | Top-5 (Val) | Skip rate |
|---|-------------|-------------|-----------|
| 15 | 28.82% | 38.93% | 43% |
| 30 | 31.53% | 41.94% | 29% |
| 50 | **34.54%** | **46.46%** | **18%** |

### 4. 종합 Ablation Study + Failure 분석
- Recall@K 진단: K=15→K=50으로 ceiling 52→71% 증가
- Failure split: 61.1% all-miss, 10.1% rank-only, 28.8% correct
- Dataset 라벨 정확도 분석: **27.5%가 8방위 기준 부정확** (대각선 36~39%)

---

## 📂 디렉토리 구조

```
suyang/
├── README.md
├── .gitignore
├── results/                              # 분석 결과 그림
│   ├── relation_angle_distribution.png   # 라벨 27.5% inconsistent 시각화
│   ├── compass_diagram.png               # 8방위 기준 정의
│   ├── axes_diagram.png                  # 각도 측정 기준
│   ├── sample_annotations.png            # consistent/inconsistent 케이스
│   ├── sample_specific_cases.png         # 대각선 라벨 부정확 예시
│   └── failure_analysis/breakdown.png    # failure category 분석
├── GroundingDINO/                        # GDINO 기반 코드
│   ├── train_lora_only.py                # Stage 1: LoRA fine-tuning
│   ├── train_srbm_bce_l3.py              # Stage 2: SRBM (BCE, L=3, K=15)
│   ├── train_srbm_K30.py                 # SRBM K=30 ablation
│   ├── train_srbm_K50.py                 # SRBM K=50 ablation
│   ├── train_srbm_on_fullft.py           # Full FT backbone + SRBM
│   ├── eval_topk.py                      # Top-1/Top-5 평가
│   ├── eval_topk_lora.py                 # LoRA only 평가
│   ├── eval_zeroshot_verify.py           # Vanilla GDINO 검증
│   ├── eval_fullft.py                    # Full FT GDINO 단독 평가
│   ├── eval_split.py                     # Relation/no-relation 분리
│   ├── eval_per_sample.py                # Per-sample 결과 저장
│   ├── analyze_failures.py               # Failure category 분석
│   ├── analyze_relation_angles.py        # 라벨 정확도 분석
│   ├── diag_recall.py                    # Recall@K 진단
│   ├── visualize_*.py                    # 시각화 스크립트들
│   └── groundingdino_modified/           # 수정한 GDINO 파일들
│       ├── groundingdino.py              # _cache 추가 (SRBM이 hidden state 접근용)
│       └── relation_v3.py                # SRBM 모듈 정의
└── AerialVG/                             # AerialVG 기반 코드
    ├── train_srbm_only.py                # AerialVG 백본 + 우리 SRBM 학습
    ├── eval_test_srbm.py                 # Test set 평가
    └── model_modified/                   # 수정한 AerialVG 파일
        ├── aerialvg.py                   # SRBM 분기 추가 (use_srbm 옵션)
        └── srbm.py                       # SRBM 모듈 (relation_v3.py 복사본)
```

---

## 🚀 환경 설정 및 재현

```bash
# Grounding DINO (IDEA-Research)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# AerialVG paper code
git clone https://github.com/IPEC-COMMUNITY/AerialVG.git
```

**GroundingDINO:**
```bash
# 학습/평가 스크립트 복사
cp suyang/GroundingDINO/*.py GroundingDINO/

# 수정된 모델 파일 적용
cp suyang/GroundingDINO/groundingdino_modified/* \
   GroundingDINO/groundingdino/models/GroundingDINO/
```

**AerialVG:**
```bash
cp suyang/AerialVG/*.py AerialVG/

# 수정된 모델 파일
cp suyang/AerialVG/model_modified/* AerialVG/model/AerialVG/
```

### 3. 데이터 준비

AerialVG 데이터셋 다운로드:
- HuggingFace: `IPEC-COMMUNITY/AerialVG`
- 또는 paper의 안내 참조

### 4. Pretrained 체크포인트

- **Vanilla GDINO**: `groundingdino_swint_ogc.pth` (IDEA-Research에서 다운로드)
- **AerialVG paper checkpoint**: `aerialvg.pth` (AerialVG repo 또는 HuggingFace)

### 5. 학습

```bash
# Stage 1: LoRA (5 epoch, ~25h)
python train_lora_only.py

# Stage 2: SRBM (15 epoch, ~20h)
python train_srbm_bce_l3.py

# AerialVG + SRBM (메인 결과)
cd AerialVG && python train_srbm_only.py
```

### 6. 평가

```bash
# Top-1/Top-5 (val set, default)
python eval_topk.py --ckpt output/srbm_bce_l3/best.pth --srbm_layers 3 --topk 15

# Test set으로 재평가
python eval_topk.py --ckpt ... --anno path/to/vg_test_odvg.jsonl

# AerialVG + SRBM test
cd AerialVG && python eval_test_srbm.py
```

---

## 🔬 SRBM 아키텍처

```
[Top-K candidates from GDINO]
       │
       ├──→ Box coords [B, K, 4] ──→ Geometry features [B, K, K, 5]
       │                                ↓
       │                          SpatialBiasMLP (5→64→1)
       │                                ↓
       │                          Spatial bias [B, K, K]   ← scalar per pair
       │                                │
       │                                │ (shared across all layers)
       │                                ↓
       └──→ Features [B, K, 256] ──→ Layer × 3:
                                      ├─ SpatialBiasSelfAttention(x, bias)
                                      ├─ CrossAttention(x, text)
                                      └─ FFN(x)
                                            ↓
                                      ContrastiveEmbed(x, text)
                                            ↓
                                      logits [B, K, max_text_len]
```

**핵심**:
- Geometry features: `[sinθ, cosθ, log(dist), log(Δw), log(Δh)]` per pair
- MLP는 1개만 사용 (모든 layer 공유)
- Bias는 self-attention logit에만 주입 (cross-attn 제외)
- 총 파라미터: **3.16M** (백본 175M의 1.8%)

---

## 📊 전체 실험 결과 (Val Set)

| Method | Trainable | Top-1 | Top-5 |
|--------|-----------|-------|-------|
| Vanilla GDINO (zero-shot) | 0 | 12.45% | 36.05% |
| Full FT GDINO (alone) | 175M | 12.32% | 49.87% |
| GDINO + LoRA | 0.9M | 22.52% | 36.73% |
| GDINO + LoRA + SRBM K=15 | +3.16M | 28.82% | 38.93% |
| GDINO + LoRA + SRBM K=30 | +3.16M | 31.53% | 41.94% |
| GDINO + LoRA + SRBM K=50 | +3.16M | 34.54% | 46.46% |
| Full FT + SRBM K=15 | +3.16M | 35.90% | TBA |
| AerialVG paper (CrossSelfRelTransformer) | 175M+ | 49.88% | 87.38% |
| **AerialVG + Our SRBM** | +3.16M | **60.81%** | **91.72%** |

---

## 📝 라이선스 / 참조

- **Grounding DINO**: Apache 2.0 (IDEA-Research)
- **AerialVG**: paper code license 참조
- **본 연구 코드**: 학술 목적 사용

---

## 👤 Contributor

- **2026 캡스톤 디자인 - 팀 44 / 변수양**
- 국민대학교
