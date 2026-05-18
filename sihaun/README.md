# AerialVG Capstone

이 레포지토리는 AerialVG baseline과 현재 사용하는 `method4`, `method6` 실험 코드만 남긴 정리본입니다. 예전 `method2`, `method3`, `method5` 계열과 데모 전용 파일은 실행 표면을 작게 유지하기 위해 제거했습니다.

## 레포 구조

- `aerialvg/`: AerialVG baseline 학습/평가 패키지
- `method4/`: frozen AerialVG detector 위에 method4 role evidence module을 붙인 실험
- `method6/`: frozen AerialVG detector 위에 method6 role evidence module을 붙인 실험과 auxiliary 진단 코드
- `groundingdino/`: 공통 GroundingDINO 유틸리티와 CUDA extension 소스
- `capstone/`: AerialVG 로컬 데이터셋 로더, checkpoint 유틸, 다운로드/검증 코드
- `checkpoints/`: 로컬 checkpoint 저장 위치. 이 폴더는 git에 올라가지 않도록 ignore됩니다.

## 환경 설정

프로젝트 환경 파일로 conda 환경을 만든 뒤 활성화합니다.

```bash
conda env create -f environment.yaml
conda activate aerialvg
```

conda를 쓰지 않는 경우 `requirements.txt`를 기준으로 설치하되, PyTorch, torchvision, CUDA, Hugging Face 관련 패키지가 현재 머신과 맞는지 확인해야 합니다.

자주 쓰는 환경 변수:

- `CUDA_VISIBLE_DEVICES=0`: 사용할 GPU 선택
- `DEVICE=cuda`: 실행 device 강제 지정. 비워두면 CUDA 사용 가능 여부에 따라 자동 선택
- `MODEL_STORAGE_DIR=./checkpoints`: checkpoint와 output root 변경
- `LOCAL_DATASET_DIR=/path/to/AerialVG`: dataset 위치 변경
- `HF_TOKEN=...`: Hugging Face token이 필요한 경우 지정
- `MSDA_DISABLE_EXT=1`: 기본값. deformable attention CUDA op 대신 PyTorch fallback 사용

## 데이터와 Checkpoint

스크립트는 먼저 `/data2/huggingface/AerialVG`에서 데이터셋을 찾습니다. 없고 `./data/AerialVG`가 있으면 로컬 복사본을 사용합니다. 데이터셋을 다운로드하거나 검증하려면:

```bash
bash download_aerialvg.sh
```

기본 checkpoint root는 `./checkpoints`입니다. 기대하는 구조는 다음과 같습니다.

```text
checkpoints/
  aerialvg/aerialvg.pth
  method4/outputs/run/best.pt
  method4/outputs/run/latest.pt
  method6/outputs/run/best.pt
  method6/outputs/run/latest.pt
```

현재 로컬 checkpoint는 `/data2/2026_capstone/sihaun/`에서 복사해 둔 상태입니다. 외부 경로를 직접 쓰고 싶으면 다음처럼 override할 수 있습니다.

```bash
MODEL_STORAGE_DIR=/data2/2026_capstone/sihaun bash infer_method6.sh
```

## Setup

새 머신에서 준비할 때는 setup script를 실행합니다. 데이터셋, AerialVG pretrained checkpoint, GroundingDINO CUDA extension, method6 custom op를 확인하고 필요한 경우 빌드합니다.

```bash
bash setup_method6.sh
```

custom op만 따로 빌드하려면:

```bash
bash build_aerialvg_ops.sh
bash build_method6_ops.sh
```

## Inference

Baseline 평가:

```bash
bash infer_aerialvg.sh
```

Method 실험 평가:

```bash
bash infer_method4.sh
bash infer_method6.sh
```

기본 checkpoint 선택 경로:

- `infer_aerialvg.sh`: `./checkpoints/aerialvg/aerialvg.pth`
- `infer_method4.sh`: `./checkpoints/method4/outputs/run/best.pt`
- `infer_method6.sh`: `./checkpoints/method6/outputs/run/best.pt`

추가 인자는 내부 Python module로 그대로 전달됩니다.

```bash
bash infer_method6.sh --batch-size 4 --top-k 15
```

## Training

Baseline 학습:

```bash
bash train_aerialvg.sh
```

Method 실험 학습:

```bash
bash train_method4.sh
bash train_method6.sh
```

`train.sh`는 method6 학습을 실행하는 wrapper입니다.

```bash
bash train.sh
```

학습 결과 checkpoint는 현재 `MODEL_STORAGE_DIR` 아래에 저장됩니다. 기본값은 `./checkpoints`입니다.

## Diagnostics

method6 auxiliary coverage 진단:

```bash
bash analyze_method6_aux.sh
```

## Sanity Check

수정 후 빠르게 확인할 때:

```bash
bash -n *.sh
find aerialvg method4 method6 capstone groundingdino -type f -name '*.py' -print0 | xargs -0 python3 -m py_compile
```

아래 smoke check는 `torch` 등 런타임 의존성이 설치된 환경에서 실행해야 합니다.

```bash
python3 -m aerialvg.eval --help
python3 -m method4.eval --help
python3 -m method6.eval --help
```
