#ifndef __KERNEL_H__
#define __KERNEL_H__

/**
 * @file kernel.h
 * @brief ESP32 kernel header file
 *
 * Interface for the minimal ESP32 kernel, including memory layout
 * symbols and the main kernel entry point.
 */

#include "types.h"

/* Memory layout symbols - defined by linker script */
extern u32 _bss_start;       /**< Start of BSS section */
extern u32 _bss_end;         /**< End of BSS section */
extern u32 _data_start;      /**< Start of data section in RAM */
extern u32 _data_end;        /**< End of data section in RAM */
extern u32 _data_load_start; /**< Start of data section in flash */
extern u32 _stack_top;       /**< Top of stack */
extern u32 _stack_bottom;    /**< Bottom of stack */

/* ESP32 specific constants */
#define ESP32_IRAM_START  0x40080000
#define ESP32_DRAM_START  0x3FFB0000
#define ESP32_STACK_START 0x3FFF0000

/* Hardware register addresses */
#define ESP32_WDT_REG   0x3FF5F048
#define ESP32_GPIO_BASE 0x3FF44000

/**
 * @brief Main kernel entry point
 *
 * Called by the bootloader after basic hardware initialization.
 * Should initialize the kernel and start the main system loop.
 */
void kernel_main(void);

/**
 * @brief Kernel panic function
 *
 * Called when the kernel encounters an unrecoverable error.
 * Should halt the system and optionally provide debugging information.
 */
void kernel_panic(const char* message);

/**
 * @brief Get kernel initialization status
 *
 * Returns whether the kernel has been initialized.
 *
 * @return 1 if initialized, 0 otherwise
 */
uint32_t kernel_is_initialized(void);

#endif /* __KERNEL_H__ */
