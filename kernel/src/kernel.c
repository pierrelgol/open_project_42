/**
 * @file kernel.c
 * @brief Minimal ESP32 kernel implementation
 *
 * Main kernel entry point and basic system initialization for ESP32.
 */

#include "kernel.h"
#include "kalloc.h"
#include "kio.h"

/** Static buffer for linker script */
static char buffer[1024];

/** Kernel initialization status */
static volatile uint32_t kernel_initialized = 0;

/**
 * @brief Initialize ESP32 hardware
 *
 * Sets up basic hardware components:
 * - Watchdog timer configuration
 * - CPU frequency setup
 * - Basic GPIO setup
 */
static void hardware_init(void) {
        /* Disable watchdog timer */
        wdt_disable();

        /* Set CPU frequency to 240MHz */
        cpu_freq_240();
}

/**
 * @brief Initialize kernel subsystems
 *
 * Sets up the basic kernel environment:
 * - Hardware initialization
 * - Memory management setup
 * - Basic system services
 */
static void kernel_init(void) {
        /* Initialize hardware */
        hardware_init();

        /* Initialize slab allocator */
        /* Use the remaining DRAM space after BSS and data sections */
        u8*   heap_start = (u8*)&_bss_end;
        usize heap_size  = (usize)ESP32_STACK_START - (usize)heap_start;

        if (kalloc_init(heap_start, heap_size) != 0) {
                kernel_panic("Failed to initialize slab allocator");
        }

        /* Initialize UART I/O subsystem */
        if (kio_init() != 0) {
                kernel_panic("Failed to initialize UART I/O subsystem");
        }

        /* Initialize UART0 for console output */
        if (kio_uart_init(UART_PORT_0) != 0) {
                kernel_panic("Failed to initialize UART0");
        }

        /* Send initial test message */
        kio_uart_send_string(UART_PORT_0, "ESP32 Kernel initialized successfully!\r\n");
        kio_uart_send_string(UART_PORT_0, "UART Test: ABCDEFGHIJKLMNOPQRSTUVWXYZ\r\n");
        kio_uart_send_string(UART_PORT_0, "UART Test: 0123456789\r\n");
        kio_uart_send_string(UART_PORT_0, "UART Test: !@#$%^&*()_+-=[]{}|;':\",./<>?\r\n");

        /* Mark kernel as initialized */
        kernel_initialized = 1;

        /* Prevent unused variable warning */
        (void)buffer;
}

/**
 * @brief Kernel panic function
 *
 * Called when the kernel encounters an unrecoverable error.
 * Halts the system and provides debugging information.
 *
 * @param message Error message to display (currently unused)
 */
void kernel_panic(const char* message) {
        /* Disable interrupts */
        __asm__("rsil a0, 15");

        /* Halt the system */
        while (1) {
                __asm__("break 1, 1");
        }

        /* Prevent unused parameter warning */
        (void)message;
}

/**
 * @brief Get kernel initialization status
 *
 * Returns whether the kernel has been initialized.
 *
 * @return 1 if initialized, 0 otherwise
 */
uint32_t kernel_is_initialized(void) {
        return kernel_initialized;
}

/**
 * @brief Main kernel entry point
 *
 * Called by the ESP32 bootloader after basic hardware initialization.
 * Initializes the kernel and starts the main system loop.
 *
 * The kernel implements a minimal system that:
 * - Initializes basic hardware
 * - Runs a main loop with power management
 * - Provides a foundation for more complex features
 */
void kernel_main(void) {
        /* Initialize kernel subsystems */
        kernel_init();

        /* Main kernel loop */
        u32 loop_counter = 0;
        while (1) {
                /* Check if kernel is properly initialized */
                if (kernel_initialized) {
                       
                        loop_counter++;
                        if (loop_counter >= 1000000) {
                                kio_uart_send_string(UART_PORT_0, "Kernel heartbeat - system running normally\r\n");
                                loop_counter = 0;
                        }

                        /* Power management - use waiti instruction for Xtensa */
                        /* This puts the CPU in a low-power state until an interrupt occurs */
                        __asm__("waiti 0");
                } else {
                        /* Kernel initialization failed - halt system */
                        kernel_panic("Kernel initialization failed");
                }
        }
}
