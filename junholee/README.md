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

Feature-DPAA Module

for each projected feature level:

│

├── (1) Down Projection

│       1×1 Conv: C → C_mid

│

├── (2) Dual-Path Depthwise Convolution

│       ├── Large-kernel branch

│       │       DWConv_large, kernel = k_L

│       │       Captures broad aerial context

│       │

│       └── Small-kernel branch

│               DWConv_small, kernel = k_S

│               Preserves local spatial detail

│

├── (3) Dual-Path Aggregation

│       z = DWConv_large(z) + DWConv_small(z)

│

├── (4) Non-linearity

│       z = GELU(z)

│

├── (5) Up Projection

│       1×1 Conv: C_mid → C

│

└── (6) Residual Connection

y = x + scale · z

## Results

| Method | Setting | Top-1 | Top-5 |
| --- | --- | --- | --- |
| GroundingDINO | Zero-Shot | 12.77 | 34.49 |
| GroundingDINO + DPAA (Ours) | Backbone Frozen | 18.00 | 44.00 |

파라미터 수 : 192,768 개

Feature-DPAA의 높은 파라미터 효율성은 bottleneck projection과 depthwise convolution 구조에서 비롯된다. 입력 feature의 channel dimension을 256에서 64로 축소한 뒤, large/small kernel convolution을 depthwise 방식으로 적용함으로써 large kernel을 사용하면서도 파라미터 증가를 최소화하였다. 또한 기존 Grounding DINO의 Detection Network는 모두 freeze하고, input projection 이후 4개의 multi-scale feature level에 삽입된 Feature-DPAA만 학습하였기 때문에 전체 학습 가능 파라미터는 192,768개, 전체 모델의 약 0.1114%에 불과하다.

## Reference

- GroundingDINO 공식: https://github.com/IDEA-Research/GroundingDINO
- AerialVG 데이터셋: Aerial Visual Grounding 벤치마크
- **Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis**

[Ziquan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+Z), [Hanruo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+H), [Siyuan Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu,+S), [Xiang Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+X), [Yanda Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng,+Y), [Gaojie Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin,+G), [Lu Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin,+L), [Lijie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+L), [Di Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+D), [Lu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+L), [Tianjin Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang,+T)
