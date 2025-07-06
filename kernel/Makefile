# Makefile for ESP32 Kernel Build System
#
# This Makefile provides targets for building, flashing, and cleaning
# the minimal ESP32 kernel implementation.
#
# Requirements:
# - ESP32 toolchain (xtensa-esp32-elf-gcc)
# - esptool.py for flashing
# - ESP32 development board

# Compiler and tools
CC = xtensa-esp32-elf-gcc
OBJCOPY = xtensa-esp32-elf-objcopy
SIZE = xtensa-esp32-elf-size

# Compiler flags for ESP32
CFLAGS = -Wall -Wextra -ffreestanding -nostdlib -Iinclude -mlongcalls -Os -mtext-section-literals
ASFLAGS = -Iinclude -mlongcalls -mtext-section-literals

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build

# Source files
SRCS = $(SRCDIR)/boot.S $(SRCDIR)/kernel.c $(SRCDIR)/std.c $(SRCDIR)/kalloc.c $(SRCDIR)/kio.c 
OBJS = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
OBJS := $(OBJS:$(SRCDIR)/%.S=$(BUILDDIR)/%.o)

# Default target
all: $(BUILDDIR) kernel.bin

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Compile C source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile assembly source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.S
	$(CC) $(ASFLAGS) -c $< -o $@

# Link object files into ELF executable
kernel.elf: $(OBJS)
	$(CC) -T linker.ld -nostdlib -o $@ $^
	$(SIZE) $@

# Convert ELF to binary for flashing
kernel.bin: kernel.elf
	$(OBJCOPY) -O binary $< $@

# Flash kernel to ESP32
# Note: Adjust port (/dev/ttyUSB0) as needed for your system
flash: kernel.bin
	esptool.py --chip esp32 --port /dev/ttyUSB0 --baud 115200 write_flash 0x10000 kernel.bin

# Monitor serial output
monitor:
	esptool.py --chip esp32 --port /dev/ttyUSB0 --baud 115200 monitor

# Flash and monitor
flash_monitor: flash
	$(MAKE) monitor

# Test: build and monitor UART output
test: all
	@echo "Building kernel and starting UART monitor..."
	@echo "Press Ctrl+A, then K to exit screen"
	screen /dev/ttyUSB0 115200

# Show size information
size: kernel.elf
	$(SIZE) $<

# Clean build artifacts
clean:
	rm -f $(BUILDDIR)/*.o *.elf *.bin
	rm -rf $(BUILDDIR)

# Full clean (including binary)
fclean: clean
	rm -f kernel.bin kernel.elf

# Rebuild from scratch
re: fclean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build kernel binary"
	@echo "  kernel.bin   - Build kernel binary"
	@echo "  flash        - Flash kernel to ESP32"
	@echo "  monitor      - Monitor serial output"
	@echo "  flash_monitor- Flash and monitor"
	@echo "  test         - Build and monitor UART output"
	@echo "  size         - Show binary size"
	@echo "  clean        - Clean build artifacts"
	@echo "  fclean       - Full clean"
	@echo "  re           - Rebuild from scratch"
	@echo "  help         - Show this help"

# Declare phony targets
.PHONY: all clean fclean re flash monitor flash_monitor test size help
