# GFIM: Gated Feature Injection for Aerial Visual Grounding


## Overview

본 연구는 드론/항공 이미지 기반 Visual Grounding의 성능 개선을 목표로 한다.

항공 이미지에서는 객체 크기가 작고 시야각이 넓어,
cross-attention 과정에서 small object에 대한 attention이 희석되는 문제가 존재한다.

이를 해결하기 위해 backbone의 low-level feature(P3)를 Feature Enhancer Layer에 반복적으로 주입하는
GFIM(Gated Feature Injection Module)을 제안한다.

GFIM은:
- Learnable Downsampling
- Adaptive Gate
- Gated Add

구조를 통해 spatial detail 정보를 선택적으로 보강한다.


## Motivation

- Swin Transformer는 stage가 깊어질수록 spatial resolution 감소
- High-level feature에서 small object detail 손실 발생
- Low-level feature(P3)에는 상대적으로 spatial information이 유지됨


## Method

본 연구는 backbone의 P3 feature를
각 feature scale(P4/P5/P6)에 맞게 변환한 뒤,
Adaptive Gate를 통해 필요한 정보만 선택적으로 fusion한다.


## Architecture

```text
TransformerEncoder (6 layers)

│
├── Save raw P3 tokens before encoder loop
│
└── for each encoder layer:
    │
    ├── (1) Fusion (BiAttentionBlock)
    │       Image ↔ Text cross-modal fusion
    │
    ├── (2) inject_p3()
    │       Inject P3 into P4/P5/P6
    │
    ├── (3) Text Self-Attention + FFN
    │
    └── (4) Deformable Self-Attention + FFN

inject_p3() 

P3 : identity connection

P4 :
    p3_down = conv[0](p3_2d)
    x_4 = x_4 + gate * p3_down

P5 :
    p3_down = conv[1](prev_down)
    x_5 = x_5 + gate * p3_down

P6 :
    p3_down = conv[2](prev_down)
    x_6 = x_6 + gate * p3_down

gate = sigmoid(W1(x_level) + W2(p3_down))
```

## Results

| Method | Setting | Top-1 | Top-5 |
|---|---|---|---|
| Grounding DINO | Zero-Shot | 12.77 | 34.49 |
| Grounding DINO + GFIM (Ours) | Backbone Frozen | **15.73** | **49.08** |

- 전체 파라미터의 약 **1.1%** 만 학습
- **5 epoch**만으로 Zero-Shot 기준 성능 향상

## References

- GroundingDINO 공식: https://github.com/IDEA-Research/GroundingDINO
- AerialVG 데이터셋: Aerial Visual Grounding 벤치마크
