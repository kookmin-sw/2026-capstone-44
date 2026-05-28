# Aerial Visual Grounding — CCM & GPQ

항공영상 Visual Grounding을 위한 Grounding-DINO 기반 모델. 객체 밀도 적응형 query 선택(**CCM**)과 query pruning(**GPQ**)을 적용·검증한 연구 코드.

- **베이스 모델**: Grounding-DINO (Swin-B + BERT)
- **데이터셋**: [AerialVG](https://huggingface.co/datasets/IPEC-COMMUNITY/AerialVG) (ICCV 2025)
- **핵심 기여**: 밀도 기반 동적 query 선택(CCM)의 항공영상 grounding 효과 입증

## 디렉토리 구조

```
aerial-grounding/
├── groundingdino/           # 모델 패키지
│   ├── config/              # 모델 config
│   ├── datasets/            # AerialVG 데이터 로더
│   ├── models/GroundingDINO/
│   │   ├── transformer.py   # 인코더/디코더 (+ GPQ 로직)
│   │   ├── groundingdino.py # 메인 모델
│   │   ├── ccm.py           # CCM (Categorical Counting Module)
│   │   ├── criterion.py     # 손실 함수 (Hungarian matcher + Focal loss)
│   │   └── backbone/        # Swin Transformer
│   └── util/
├── train/                   # 학습 스크립트
│   ├── train_base.py        # Baseline (고정 900 query)
│   ├── train_ccm.py         # + CCM (밀도 적응형 query)
│   └── train_gpq_ccm.py     # + CCM + GPQ (query pruning)
└── eval/                    # 평가 스크립트
    └── eval_topk_acc.py     # AerialVG 공식 프로토콜 (Top-1/Top-5 Acc)
```

## 설치

```bash
# 1. 환경
conda create -n vision_env python=3.10
conda activate vision_env
pip install -r requirements.txt

# 2. CUDA 커스텀 연산 컴파일 (Deformable Attention)
cd groundingdino/models/GroundingDINO/csrc
# (또는 ops 디렉토리의 setup.py) python setup.py build install

# 3. 사전학습 가중치 다운로드 → groundingdino/weights/ 에 배치
#    groundingdino_swinb_cogcoor.pth (Grounding-DINO 공식 repo)
```

## 학습

```bash
# Baseline
python train/train_base.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --pretrained_weights groundingdino/weights/groundingdino_swinb_cogcoor.pth \
    --output_dir outputs/baseline --num_epochs 12 --batch_size 8

# CCM
python train/train_ccm.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --pretrained_weights groundingdino/weights/groundingdino_swinb_cogcoor.pth \
    --output_dir outputs/ccm --num_epochs 12 --batch_size 8

# CCM + GPQ
python train/train_gpq_ccm.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --pretrained_weights groundingdino/weights/groundingdino_swinb_cogcoor.pth \
    --output_dir outputs/gpq_ccm --num_epochs 12 --batch_size 8
```

## 평가

```bash
# AerialVG 공식 Top-1/Top-5 Accuracy (@ IoU > 0.5)
python eval/eval_topk_acc.py --checkpoint outputs/ccm/latest.pth --split test
python eval/eval_topk_acc.py --checkpoint outputs/baseline/latest.pth --split test --no_ccm
```

## 주요 결과 (AerialVG test, 12 epoch)

| 모델 | Top-1 Acc | Top-5 Acc |
|---|:---:|:---:|
| Baseline | 16.41% | 38.39% |
| **+ CCM** | **17.57%** | **44.97%** |
| + CCM+GPQ | 17.32% | 40.27% |

- **CCM**: Baseline 대비 Top-1 +1.16%p, Top-5 +6.58%p
- **GPQ**: AerialVG 저밀도 특성으로 query cap 미발동 → 효율 개선 효과 제한적 (분석적 negative result)

## 참고

- Grounding-DINO: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- AerialVG: [Ideal-ljl/AerialVG](https://github.com/Ideal-ljl/AerialVG)
- CCM 아이디어: DQ-DETR
- GPQ 아이디어: [Redundant Queries in DETR-Based 3D Detection](https://arxiv.org/abs/2412.02054)
