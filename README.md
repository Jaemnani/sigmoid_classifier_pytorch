# Sigmoid Classifier (PyTorch)

This repository is a PyTorch implementation of an image classifier based on the Sigmoid activation function. It includes features like Adaptive Cross Entropy Loss, ~~Class Activation Map (CAM) visualization~~, and Real-time Training Loss Plot.

## Features

- **PyTorch Implementation**: Ported from an existing framework to PyTorch.
- **Adaptive Cross Entropy**: Adjusts loss function based on training difficulty.
- **Label Smoothing**: Supports label smoothing to prevent overfitting.
- **Efficient Architecture**: Supports lightweight model architecture (Depthwise Separable Convolution).
- **CAM (Class Activation Map)**: Visualizes which parts of the image the model is focusing on (viewable during training).
- **Live Plot**: Visualizes training progress (Loss) in a real-time graph. Optimized with fixed validation batch for speed.
- **Early Stopping**: Automatically stops training if validation accuracy doesn't improve for a specified number of checks.
- **Data Augmentation**: Robust data augmentation using the `albumentations` library (includes CoarseDropout/Cutout).

## Installation

Install the required libraries.

```bash
pip install -r requirements.txt
```

**Requirements:**
- python 3.8+
- torch
- torchvision
- opencv-python
- tqdm
- matplotlib
- albumentations

## Dataset Preparation

Datasets should be organized in the following directory structure:

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

- If an `unknown` folder exists, images in that folder can be included as an 'unknown' class for training/evaluation.
- Unique Feature: 'Unknown' Class Handling If an unknown folder exists in the dataset, images within it are used to train the model to output all zeros (e.g., [0, 0, ..., 0]). Unlike standard classifiers, the 'unknown' category does not increase the num_classes. Instead, the model learns to suppress activations for all known classes when encountering unknown data.

## Usage

### 1. Training

Run `train.py` to start training. Training configurations can be modified directly in the `SigmoidClassifier` constructor arguments within `train.py`.

```python
# Example configuration in train.py
classifier = SigmoidClassifier(
    train_image_path='./train_data/mnist/train',
    validation_image_path='./train_data/mnist/validation',
    model_name='mnist',
    input_shape=(28, 28, 1), # (Height, Width, Channel)
    lr=0.001,
    batch_size=256,
    iterations=10000,
    architecture='original', # 'original' or 'efficient'
    lr_policy='step', # 'step', 'step2', 'cosine', 'onecycle', 'constant'
    early_stopping_patience=10, 
    ...
)
```

Run:
```bash
python train.py
```

**Key Parameters:**
- `input_shape`: Input image dimensions (H, W, C). (e.g., 28x28 grayscale is (28, 28, 1))
- `lr`: Learning Rate.
- `lr_policy`: Learning rate scheduler approach ('step', 'step2', 'cosine', 'onecycle', 'constant').
- `architecture`: Model architecture ('original' for standard Conv2D, 'efficient' for Depthwise Separable Conv).
- `iterations`: Total number of training iterations.
- `warm_up`: Ratio of total iterations for learning rate warm-up.
- `alpha`, `gamma`: Hyperparameters for Adaptive Cross Entropy Loss.
- `aug_brightness`, `aug_contrast`, `aug_rotate`, `aug_h_flip`: Data augmentation parameters.
- `checkpoint_interval`: Interval for saving models and evaluation.
- `early_stopping_patience`: Number of evaluation intervals to wait for improvement before early stopping (0 to disable).
- `show_class_activation_map`: If set to `True`, displays CAM visualization window during training.
- `show_live_plot`: If set to `True`, displays Loss graph window during training.

### 2. Evaluation

To evaluate the trained model on the validation dataset, use the `--evaluate` option.

```bash
python train.py --evaluate --dataset validation
```
- `--model`: (Optional) Specify a specific checkpoint model path to load. If not specified, it may follow settings in the code or default paths.

### 3. Automatic Classification & Inference

You can automatically classify images in a folder using `auto_classification.py`. Modify the path settings in the `main()` function to match your environment before running.

```python
# Inside auto_classification.py
def main():
    model_path = 'results/train/mnist/best.pt'
    img_path = '/path/to/images'
    # num_classes, input_shape, and architecture must match training settings.
    auto_classification(
        model_path, 
        img_path, 
        save_path, 
        num_classes=10, 
        input_shape=(1, 28, 28),
        architecture='original',
        save_with_score=False, 
        unknown_threshold=0.5
    )
```

Run:
```bash
python auto_classification.py
```

## Notes

- It automatically uses a CUDA-enabled GPU if available. For Mac, support for MPS (Metal Performance Shaders) is included in the code.
- Training results (model checkpoints, graphs, etc.) are saved in the `results/train/` directory.
- Auto classification uses multi-threading for faster image loading.
