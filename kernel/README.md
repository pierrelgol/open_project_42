# ESP32 Kernel

A minimal kernel for the ESP32 microcontroller. Part of the computer vision project for embedded obstacle detection.

## Overview

This kernel provides a basic foundation for running the CNN object detection model on ESP32 hardware. It's lightweight and efficient, good for real-time processing with limited resources.

## Project structure

```
kernel/
├── src/                    # Source files
│   ├── boot.S             # Assembly boot code
│   ├── kernel.c           # Main kernel implementation
│   └── linker.ld          # Linker script
├── include/               # Header files
│   └── kernel.h           # Kernel interface
├── Makefile               # Build system
└── README.md              # This file
```

## Features

- **Minimal Footprint**: Lightweight kernel suitable for ESP32's limited resources
- **Bare Metal**: Runs without an operating system for maximum performance
- **Modular Design**: Clean separation between boot code and kernel logic
- **Extensible**: Designed to be easily extended with additional functionality

## Requirements

- **Hardware**: ESP32 development board
- **Toolchain**: ESP32 GCC toolchain (`xtensa-esp32-elf-gcc`)
- **Tools**: `esptool.py` for flashing

## Building

### Prerequisites

1. Install ESP32 toolchain:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install gcc-xtensa-esp32-elf
   
   # Or download from Espressif
   # https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/
   ```

2. Install esptool:
   ```bash
   pip install esptool
   ```

### Build commands

```bash
# Build kernel
make all

# Clean build artifacts
make clean

# Full clean
make fclean

# Rebuild from scratch
make re
```

## Flashing

```bash
# Flash to ESP32 (adjust port as needed)
make flash

# Manual flashing with custom port
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash 0x10000 kernel.bin
```

## Architecture

### Boot process

1. **Hardware Reset**: ESP32 boot ROM initializes basic hardware
2. **Boot Code** (`boot.S`): Sets up stack and calls kernel
3. **Kernel Entry** (`kernel_main()`): Main kernel initialization and loop

### Current implementation

The current kernel is minimal and includes:

- Basic hardware initialization
- Stack setup
- Main kernel loop with power management
- Placeholder for future extensions

## Future extensions

The kernel is designed to be extended with:

- **CNN Integration**: Loading and running the trained model
- **Camera Interface**: Reading from ESP32-CAM or similar
- **Communication**: WiFi/BLE for data transmission
- **Real-time Processing**: Optimized inference pipeline
- **Power Management**: Dynamic frequency scaling and sleep modes

## Development

### Adding new features

1. **Header Files**: Add function declarations to `include/kernel.h`
2. **Implementation**: Add code to `src/kernel.c`
3. **Build**: Update Makefile if new source files are added
4. **Test**: Build and flash to verify functionality

### Debugging

- Use `esptool.py` to monitor serial output
- Add debug prints to kernel code
- Use ESP32's built-in debugging capabilities

## Troubleshooting

### Common issues

1. **Build Errors**: Ensure ESP32 toolchain is properly installed
2. **Flash Errors**: Check USB connection and port settings
3. **Runtime Issues**: Verify memory layout and stack setup

### Getting help

- Check ESP32 documentation: https://docs.espressif.com/
- Review linker script for memory layout issues
- Verify toolchain installation and PATH settings

## License

See the main project LICENSE file for details. 
