# AerialVG Visual Grounding with Dual-Path Aerial Adapter (DPAA)

본 연구는 항공 이미지 Visual Grounding (AerialVG) 데이터셋에서 GroundingDINO를 보완하고 성능 향상을 위한  **Dual-Path Aerial Adapter(DPAA)**  제안합니다.

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

## Parameter Efficiency

Feature-DPAA is highly parameter-efficient because it combines **bottleneck projection** with **depthwise convolution**.

First, the input feature channel dimension is reduced from `C = 256` to `C_mid = 64` using a 1x1 bottleneck projection. Then, both the large-kernel and small-kernel convolutions are applied in a depthwise manner. This design allows Feature-DPAA to use a large receptive field while minimizing the number of additional trainable parameters.

### Parameters

| Component | Calculation | Parameters |
|---|---:|---:|
| Down 1x1 Conv | `256 x 64 + 64` | 16,448 |
| Large DWConv 15x15 | `64 x 15 x 15 + 64` | 14,464 |
| Small DWConv 3x3 | `64 x 3 x 3 + 64` | 640 |
| Up 1x1 Conv | `64 x 256 + 256` | 16,640 |
| **Total per module** | - | **48,192** |

Since Feature-DPAA is inserted into four multi-scale feature levels, the total number of trainable parameters is:

**48,192 x 4 = 192,768 trainable parameters**

The full model size with Feature-DPAA is summarized below.

| Item | Value |
|---|---:|
| Total parameters | 173,032,450 |
| Trainable parameters | 192,768 |

---

## References

- **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**
  - Official Repository: https://github.com/IDEA-Research/GroundingDINO

- **AerialVG: A Challenging Benchmark for Aerial Visual Grounding by Exploring Positional Relations**
  - Aerial visual grounding benchmark used for evaluation.

- **Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis**
  - Ziquan Zhu, Hanruo Zhu, Siyuan Lu, Xiang Li, Yanda Meng, Gaojie Jin, Lu Yin, Lijie Hu, Di Wang, Lu Liu, Tianjin Huang.
  - ICLR 2026.
