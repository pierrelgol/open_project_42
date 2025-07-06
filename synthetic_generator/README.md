# Synthetic Data Generator for Object Detection

A Zig-based tool that creates labeled image datasets for training object detection models. Generates scenes with geometric shapes, draws bounding boxes around them, takes screenshots, and produces YOLO-format label files.

## Features

- **Multiple Shape Types**: Generates circles, rectangles, and triangles
- **Random Scene Generation**: Creates scenes with 1-10 randomly positioned shapes
- **Bounding Box Visualization**: Draws red bounding boxes around each shape
- **YOLO Format Labels**: Outputs normalized coordinates in YOLO format
- **High-Quality Screenshots**: Captures scenes as PNG images
- **Configurable Parameters**: Adjustable image size, shape counts, and sizes

## Requirements

- Zig compiler (version 0.15.0 or later)
- raylib-zig dependency (automatically managed)

## Building and running

### Build the project:
```bash
zig build
```

### Run the generator:
```bash
zig build run
```

### Run tests:
```bash
make test
```

## Output structure

The generator creates a `dataset` directory with the following structure:

```
dataset/
├── images/
│   ├── sample_0000.png
│   ├── sample_0001.png
│   └── ...
└── labels/
    ├── sample_0000.txt
    ├── sample_0001.txt
    └── ...
```

## Label format

Each label file contains YOLO-format annotations with normalized coordinates:

```
<class_id> <x_center> <y_center> <width> <height>
```

### Class IDs:
- `0`: Circle
- `1`: Rectangle  
- `2`: Triangle

### Example label file (sample_0000.txt):
```
0 0.234567 0.345678 0.123456 0.123456
1 0.567890 0.456789 0.234567 0.234567
2 0.789012 0.678901 0.345678 0.345678
```

## Configuration

You can modify the following constants in `src/main.zig`:

```zig
const SCREEN_WIDTH = 800;           // Image width
const SCREEN_HEIGHT = 600;          // Image height
const MAX_SHAPES_PER_SCENE = 10;    // Maximum shapes per scene
const MIN_SHAPE_SIZE = 30;          // Minimum shape size
const MAX_SHAPE_SIZE = 100;         // Maximum shape size
```

## Dataset generation process

1. **Scene Generation**: Creates a random scene with 1-10 shapes
2. **Shape Rendering**: Draws each shape with random colors
3. **Bounding Box Drawing**: Adds red bounding boxes around shapes
4. **Screenshot Capture**: Takes a screenshot of the scene
5. **Label Generation**: Creates YOLO-format labels with normalized coordinates
6. **File Saving**: Saves both image and label files

## Usage for machine learning

The generated dataset is ready for training object detection models:

- **Images**: PNG format, 800x600 pixels
- **Labels**: YOLO format with normalized coordinates
- **Classes**: 3 shape types (circle, rectangle, triangle)
- **Variability**: Random positions, sizes, colors, and counts

## Customization

### Adding new shape types

1. Add new shape type to the `ShapeType` enum
2. Implement drawing logic in the `Scene.draw()` method
3. Update the `boundingBoxToYOLO()` function with new class ID
4. Add the shape type to `randomShapeType()`

### Modifying scene generation

- Adjust shape positioning logic in `generateScene()`
- Modify color generation in `randomColor()`
- Change background colors in `Scene.init()`

## Performance

- Generates 100 samples by default
- Each sample takes ~10ms to generate
- Total generation time: ~1 second
- Memory efficient with proper cleanup

## Troubleshooting

### Common issues:

1. **Build errors**: Ensure Zig version is 0.15.0+
2. **Missing dependencies**: Run `zig build` to fetch raylib-zig
3. **Permission errors**: Ensure write permissions in the project directory
4. **Display issues**: The generator requires a display for raylib rendering

### Debug output:

The generator provides progress information:
```
Generating 100 synthetic dataset samples...
Generated sample 0: sample_0000.png
Generated sample 1: sample_0001.png
...
Dataset generation complete! Check the 'dataset' directory.
```

## License

This project uses the MIT license. See LICENSE file for details. 