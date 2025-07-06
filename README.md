# Free Project - Computer Vision and Embedded Systems

This repo has a computer vision project with CNN object detection, an embedded kernel for ESP32, and a synthetic data generator.

## What it does

- Detects obstacles and large objects in front of the user
- Covers a 120° field of view
- Works at 15-30m range with ~50-100 MOA resolution
- Gives real-time haptic feedback based on 3D awareness
- Low latency (<100ms) and runs at 10-15 FPS
- Battery lasts all day, ideally a full work week

## Project structure

```
free-projet/
├── cnn/                    # CNN object detection
│   └── proto_v1/          # First prototype with PyTorch
│       ├── model.py       # ResNet18 detection model
│       ├── train.py       # Training script
│       ├── main.py        # Inference and visualization
│       ├── data/          # Data loading and augmentation
│       └── README.md      # CNN docs
├── kernel/                # ESP32 kernel
│   ├── src/               # Kernel source
│   ├── include/           # Headers
│   └── Makefile          # Build system
└── synthetic_generator/   # Zig synthetic data generator
    ├── src/               # Generator source
    ├── build.zig          # Build configuration
    └── README.md          # Generator docs
```

## Quick start

### CNN Object Detection
```bash
cd cnn/proto_v1
make activate          # Activate virtual environment
make train             # Train the model
make inference         # Run inference on sample image
```

### Embedded Kernel
```bash
cd kernel
make all               # Build kernel
make flash             # Flash to ESP32 (needs esptool)
```

### Synthetic Data Generator
```bash
cd synthetic_generator
zig build run          # Generate synthetic dataset
make test              # Run tests
```

## Requirements

- **CNN**: Python 3.8+, PyTorch, CUDA (optional)
- **Kernel**: ESP32 toolchain, esptool.py
- **Generator**: Zig compiler (0.15.0+)

## Docs

- [CNN Documentation](cnn/proto_v1/README.md) - Object detection guide
- [Kernel Documentation](kernel/) - Embedded system stuff
- [Generator Documentation](synthetic_generator/README.md) - Synthetic data generation

## License

See [LICENSE](LICENSE) file.
