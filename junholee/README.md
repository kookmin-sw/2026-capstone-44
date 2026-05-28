## Overview

본 연구는 **드론/항공 이미지 기반 Visual Grounding의 성능 개선**을 목표로 한다.

항공 이미지에서는 객체 크기가 작고 장면의 시야각이 넓어, 기존 Grounding DINO가 작은 객체의 위치 정보를 정확히 보존하면서도 넓은 spatial context를 함께 반영하는 데 한계가 존재한다.

이를 해결하기 위해 Grounding DINO의 **Feature Enhancer 입력 전 multi-scale visual feature**에 적용되는 **Feature-DPAA(Dual-Path Aerial Adapter)**를 제안한다.

Feature-DPAA

Bottleneck Projection
Large-kernel Depthwise Convolution
Small-kernel Depthwise Convolution
Residual Feature Adaptation

구조를 통해 항공 이미지의 **넓은 문맥 정보**와 **작은 객체의 세부 위치 정보**를 동시에 보강한다.

## Motivation

항공 이미지에서는 객체가 작고 배경 영역이 넓어 small object localization이 어렵다.

Swin backbone과 multi-scale projection을 거치며 작은 객체의 spatial detail이 약화될 수 있다.

Caption에는 left, right, near, above와 같은 공간 관계 표현이 포함되므로 넓은 context 이해가 필요하다.

## Method

본 연구는 Grounding DINO의 **input projection 이후 생성되는 multi-scale visual feature**에 DKA 기반 **Feature-DPAA(Dual-Path Aerial Adapter)**를 삽입하여, 항공 이미지에 특화된 visual feature를 보정한다.

Feature-DPAA는 각 feature scale에 대해 입력 feature를 bottleneck projection으로 축소한 뒤, **large-kernel depthwise convolution**과 **small-kernel depthwise convolution**을 병렬로 적용한다.

Large-kernel branch는 넓은 항공 이미지 문맥을 포착하고, small-kernel branch는 작은 객체의 세부 위치 정보를 보존한다.

두 branch의 출력을 합산한 뒤 GELU와 up projection을 거쳐 원래 feature에 residual 방식으로 더함으로써, 기존 Grounding DINO의 feature representation을 항공 이미지에 맞게 보강한다

## Repository
junholee/
├── feature_dpaa_adapter.py
├── train/
│   └── train_aerial.py
├── groundingdino/
│   └── groundingdino.py
├── validation/
│   └── val_aerial.py
│   └── visualize_aerial.py
├── environment.yaml
├── README.md
└── requirements.txt

## Architecture

## Architecture

```text
Grounding DINO with Feature-DPAA

Input Image + Text Query
        |
        v
Frozen Swin Image Backbone
        |
        v
Frozen Input Projection
        |
        +-- input_proj[0] -> Feature-DPAA
        |
        +-- input_proj[1] -> Feature-DPAA
        |
        +-- input_proj[2] -> Feature-DPAA
        |
        +-- input_proj[3] -> Feature-DPAA
        |
        v
Frozen Feature Enhancer / Transformer
        |
        v
Frozen Decoder
        |
        v
Frozen Prediction Head
        |
        v
Bounding Box Prediction
```

```text
Feature-DPAA Module

Input feature x: [B, C, H, W]

        x
        |
        v
+----------------------+
| Down Projection      |
| 1x1 Conv: C -> C_mid |
+----------------------+
        |
        v
        z
        |
        +-------------------------------+
        |                               |
        v                               v
+----------------------+        +----------------------+
| Large-kernel DWConv  |        | Small-kernel DWConv  |
| kernel = k_L         |        | kernel = k_S         |
| broad aerial context |        | fine spatial detail  |
+----------------------+        +----------------------+
        |                               |
        +---------------+---------------+
                        |
                        v
+----------------------+
| Path Aggregation     |
| z_L + z_S            |
+----------------------+
                        |
                        v
+----------------------+
| GELU                 |
+----------------------+
                        |
                        v
+----------------------+
| Up Projection        |
| 1x1 Conv: C_mid -> C |
+----------------------+
                        |
                        v
+----------------------+
| Residual Update      |
| y = x + scale * Up(z)|
+----------------------+
```

The mathematical formulation is:

```text
z = Down(x)

z_L = DWConv_large(z)

z_S = DWConv_small(z)

z_A = GELU(z_L + z_S)

y = x + scale * Up(z_A)
```

where:

- `x`: input projected visual feature
- `Down`: 1x1 bottleneck projection
- `DWConv_large`: large-kernel depthwise convolution
- `DWConv_small`: small-kernel depthwise convolution
- `Up`: 1x1 channel restoration projection
- `scale`: residual scaling factor


## Results

| Method | Setting | Top-1 | Top-5 |
| --- | --- | --- | --- |
| GroundingDINO | Zero-Shot | 12.77 | 34.49 |
| GroundingDINO + DPAA (Ours) | Backbone Frozen | 18.00 | 44.00 |

## Parameters
Down 1×1 Conv        : 256 × 64 + 64 = 16,448
Large DWConv 15×15   : 64 × 15 × 15 + 64 = 14,464
Small DWConv 3×3     : 64 × 3 × 3 + 64 = 640
Up 1×1 Conv          : 64 × 256 + 256 = 16,640

Total per module     : 48,192

Total parameter : 192,768 

Feature-DPAA의 높은 파라미터 효율성은 bottleneck projection과 depthwise convolution 구조에서 비롯된다. 입력 feature의 channel dimension을 256에서 64로 축소한 뒤, large/small kernel convolution을 depthwise 방식으로 적용함으로써 large kernel을 사용하면서도 파라미터 증가를 최소화하였다. 또한 기존 Grounding DINO의 Detection Network는 모두 freeze하고, input projection 이후 4개의 multi-scale feature level에 삽입된 Feature-DPAA만 학습하였기 때문에 전체 학습 가능 파라미터는 192,768개, 전체 모델의 약 0.1114%에 불과하다.

## Reference

- GroundingDINO 공식: https://github.com/IDEA-Research/GroundingDINO
- AerialVG 데이터셋: Aerial Visual Grounding 벤치마크
- **Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis, Ziquan Zhu, Hanruo Zhu et al, ICLR 2026 **
