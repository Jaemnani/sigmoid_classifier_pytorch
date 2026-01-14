# Sigmoid Classifier (PyTorch)

이 저장소는 Sigmoid 활성화 함수 기반의 이미지 분류기를 PyTorch로 구현한 프로젝트입니다. Adaptive Cross Entropy Loss, ~~Class Activation Map (CAM) 시각화, 실시간 학습 그래프~~ 등 다양한 기능을 포함하고 있습니다.

## 주요 기능 (Features)

- **PyTorch 구현**: 기존 프레임워크에서 PyTorch로 포팅되었습니다.
- **Adaptive Cross Entropy**: 학습 난이도에 따라 손실 함수를 조절하는 기능.
- **Label Smoothing**: 과적합 방지를 위한 레이블 스무딩 지원.

- ~~**CAM (Class Activation Map)**: 모델이 이미지의 어느 부분을 보고 판단했는지 시각화 (학습 중 확인 가능).~~
- ~~**Live Plot**: 학습 진행 상황(Loss)을 실시간 그래프로 시각화.~~
- **Data Augmentation**: `albumentations` 라이브러리를 활용한 강력한 데이터 증강.

## 설치 (Installation)

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

**요구 사항:**
- python 3.8+
- torch
- torchvision
- opencv-python
- tqdm
- matplotlib
- albumentations

## 데이터셋 준비 (Dataset Preparation)

데이터셋은 다음과 같은 디렉토리 구조로 정리해야 합니다.

```
train_data/
  └── dataset_name/
      ├── train/
      │   ├── class_1/
      │   │   ├── image1.jpg
      │   │   └── ...
      │   └── class_2/
      └── validation/
          ├── class_1/
          └── class_2/
```

- `unknown` 폴더가 있을 경우, 해당 폴더의 이미지는 unknown 클래스로 학습/평가에 반영될 수 있습니다.

## 사용 방법 (Usage)

### 1. 학습 (Training)

`train.py` 파일을 실행하여 학습을 시작합니다. 학습 설정은 `train.py` 파일 내의 `SigmoidClassifier` 생성자 인자에서 직접 수정할 수 있습니다.

```python
# train.py 내부 설정 예시
classifier = SigmoidClassifier(
    train_image_path='./train_data/mnist/train',
    validation_image_path='./train_data/mnist/validation',
    model_name='mnist',
    input_shape=(28, 28, 1), # (Height, Width, Channel)
    lr=0.001,
    batch_size=256,
    iterations=1000,
    ...
)
```

실행:
```bash
python train.py
```

**주요 파라미터:**
- `input_shape`: 입력 이미지 크기 (H, W, C). (예: 28x28 흑백은 (28, 28, 1))
- `lr`: 학습률 (Learning Rate).
- `iterations`: 총 학습 반복 횟수.
- `checkpoint_interval`: 모델 저장 및 평가 주기.
- ~~`show_class_activation_map`: `True`로 설정 시 학습 중 CAM 시각화 창 표시.~~
- ~~`show_live_plot`: `True`로 설정 시 Loss 그래프 창 표시.~~

### 2. 평가 (Evaluation)

학습된 모델을 검증 데이터셋으로 평가하려면 `--evaluate` 옵션을 사용합니다.

```bash
python train.py --evaluate --dataset validation
```
- `--model`: (선택사항) 특정 체크포인트 모델 경로를 지정하여 로드할 수 있습니다. 지정하지 않으면 코드 내 설정이나 기본 경로를 따를 수 있습니다.

### 3. 자동 분류 및 추론 (Inference)

`auto_classification.py`를 사용하여 폴더 내의 이미지를 자동으로 분류할 수 있습니다. `main()` 함수 내의 경로 설정을 환경에 맞게 수정한 후 실행하세요.

```python
# auto_classification.py 내부
def main():
    model_path = 'checkpoint/mnist/best.pt'
    img_path = '/path/to/images'
    # num_classes와 input_shape는 학습 시 설정과 동일해야 합니다.
    auto_classification(model_path, img_path, num_classes=1000, input_shape=(3, 64, 64))
```

실행:
```bash
python auto_classification.py
```

## 참고 사항

- CUDA 사용 가능한 GPU가 있는 경우 자동으로 사용하며, Mac의 경우 MPS(Metal Performance Shaders)를 지원하도록 코드에 포함되어 있습니다.
- 학습 결과(모델 체크포인트, 그래프 등)는 `checkpoint/` 디렉토리에 저장됩니다.
