# CNN Object Detection Model

A simple CNN object detection model using PyTorch. Uses ResNet18 as the backbone and outputs bounding boxes.

## Project structure

```
cnn/proto_v1/
├── train.py              # Training script with mixed precision
├── main.py               # Inference and visualization
├── model.py              # CNN model with ResNet18 backbone
├── loss.py               # Detection loss function
├── utils.py              # Utility functions
├── data/                 # Data loading and augmentation
│   ├── loader.py         # COCO128 dataset loader
│   ├── augmentation.py   # Data augmentation
│   └── extract_coco128.py # Dataset extraction
└── Makefile              # Build and training commands
```

## Features

- Mixed precision training for faster training
- Data augmentation to improve generalization
- Simple CLI for training and inference
- GPU support with CPU fallback
- Modular design with separate modules

## Quick start

### 1. Setup

```bash
# Activate virtual environment
make activate

# Extract dataset (if needed)
python data/extract_coco128.py
```

### 2. Training

```bash
# Train with optimized settings
make train

# Train with basic settings
make train-basic

# Custom training
python train.py --epochs 100 --batch-size 32 --learning-rate 1e-3
```

### 3. Inference

```bash
# Run inference on an image
python main.py inference path/to/image.jpg --model-path checkpoints/model_final.pt

# Visualize predictions
python main.py visualize --model-path checkpoints/model_final.pt --output-path outputs/visualization.jpg
```

## Makefile commands

- `make train` - Train with optimized settings
- `make train-basic` - Train with basic settings (no mixed precision, no augmentation)
- `make inference` - Run inference on sample image
- `make visualize` - Create visualization of predictions
- `make clean` - Clean generated files
- `make fclean` - Full clean (including dataset)
- `make activate` - Show virtual environment activation command

## Model architecture

- **Backbone**: ResNet18 modified for grayscale input (1 channel)
- **Detection Head**: Custom head for bounding box regression
- **Output**: 5-channel output (x_rel, y_rel, w, h, confidence) per grid cell
- **Grid Size**: 13x13 (for 416x416 input, 32x downsampling)
- **Input**: Grayscale images resized to 416x416
- **Output Format**: YOLO-style predictions with relative coordinates

### How it works

The model uses ResNet18 as the backbone with these changes:
1. **Input Layer**: Modified first conv layer to accept 1-channel grayscale input
2. **Downsampling**: 32x total downsampling (416 → 13x13 grid)
3. **Detection Head**: 3 conv layers (512→256→128→64→5) with batch norm and ReLU
4. **Output**: Sigmoid activation for all outputs (coordinates and confidence)

### Grid system

- **Grid Size**: 13×13 (416÷32 = 13)
- **Cell Predictions**: Each grid cell predicts 5 values:
  - `x_rel`: Relative x offset from cell center (0-1)
  - `y_rel`: Relative y offset from cell center (0-1) 
  - `w`: Normalized width (0-1)
  - `h`: Normalized height (0-1)
  - `confidence`: Objectness score (0-1)

## Training features

- **Mixed Precision**: Automatic mixed precision training for faster training and lower memory usage
- **Data Augmentation**: Geometric and photometric transformations
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Optimized DataLoader**: Multi-worker loading with pin memory and prefetching

## Configuration

Key training parameters can be adjusted in the Makefile or via command-line arguments:

- `BATCH_SIZE`: Training batch size (default: 16)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 50)
- `NUM_WORKERS`: DataLoader workers (default: 8, auto-detected)

## Output

- **Checkpoints**: Saved every 10 epochs in `checkpoints/` directory
- **Final Model**: `checkpoints/model_final.pt`
- **Visualizations**: Saved to `outputs/` directory

## Performance

- **Memory Usage**: ~2GB GPU memory with batch size 16
- **Training Time**: ~30-60 minutes on GTX 1080 for 50 epochs
- **Inference Speed**: ~10-15 FPS on GPU, ~2-3 FPS on CPU
- **Model Size**: ~45MB (ResNet18 + detection head) 
