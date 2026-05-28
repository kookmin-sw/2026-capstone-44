# DPAA: Dual-Path Aerial Adapter for Aerial Visual Grounding

## Overview

This project aims to improve **Aerial Visual Grounding** performance on drone/aerial images using **Grounding DINO** as the baseline model.

Aerial images have two major challenges:

1. **Small object localization**  
   Objects in aerial images are often very small due to high-altitude viewpoints.

2. **Wide spatial context understanding**  
   Aerial captions often include spatial relations such as `left`, `right`, `near`, `above`, and `next to`, requiring broad scene-level context.

To address these issues, we propose **Feature-DPAA (Dual-Path Aerial Adapter)**, a lightweight parameter-efficient adapter inserted into the multi-scale visual features of Grounding DINO.

Feature-DPAA enhances aerial visual features using:

- Bottleneck Projection
- Large-kernel Depthwise Convolution
- Small-kernel Depthwise Convolution
- Dual-path Feature Aggregation
- Residual Feature Adaptation

The original Grounding DINO Detection Network is frozen, and only the newly inserted Feature-DPAA parameters are trained.

---

## Motivation

Grounding DINO is pretrained mainly on general natural images.  
When it is directly applied to aerial images, performance can degrade due to domain shift.

In aerial visual grounding:

- Objects are small and easily lose spatial details during feature extraction.
- Wide-view scenes require understanding of global spatial context.
- Captions often describe objects using spatial relationships.
- Full fine-tuning of the Detection Network is not allowed in this project setting.

Therefore, a lightweight adapter is needed to adapt visual features to the aerial domain while keeping the original Grounding DINO model frozen.

---

## Method

Feature-DPAA is inserted after the input projection layers of Grounding DINO.

Grounding DINO uses four multi-scale visual feature levels.  
Therefore, Feature-DPAA is applied to:

```text
input_proj[0] → Feature-DPAA
input_proj[1] → Feature-DPAA
input_proj[2] → Feature-DPAA
input_proj[3] → Feature-DPAA
