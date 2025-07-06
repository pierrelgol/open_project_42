/**
 * @file kio.c
 * @brief ESP32 UART I/O implementation
 *
 * This file contains the implementation of UART communication functions
 * for the ESP32 embedded system, following the reference implementation.
 */

#include "kio.h"
#include "kernel.h"

/* UART base addresses array */
static volatile u32* uart_bases[UART_PORT_MAX] = {
    REG(ESP32_UART0),
    REG(ESP32_UART1),
    REG(ESP32_UART2),
};

/* UART initialization status */
static b8 uart_initialized[UART_PORT_MAX] = {
    false,
    false,
    false,
};

/* UART GPIO pins for ESP32 */
#define UART0_TX_PIN 1 /* GPIO1 */
#define UART0_RX_PIN 3 /* GPIO3 */

/**
 * @brief Perform NOP operations for delay
 *
 * @param count Number of NOP operations to perform
 */
static inline void spin(volatile unsigned long count) {
        while (count--) asm volatile("nop");
}

/**
 * @brief Disable watchdog timer
 *
 * Disables the ESP32 watchdog timer to prevent system resets.
 * This function should be called early in system initialization.
 */
void wdt_disable(void) {
        REG(ESP32_RTCCNTL)[41] = 0x50d83aa1; /* Disable write protection */
        REG(ESP32_RTCCNTL)[40] |= BIT(31);   /* Feed watchdog */
        REG(ESP32_RTCCNTL)[35]     = 0;      /* Disable RTC WDT */
        REG(ESP32_TIMERGROUP0)[18] = 0;      /* Disable task WDT */
}

/**
 * @brief Set CPU frequency to 240MHz
 *
 * Configures the ESP32 CPU to run at 240MHz frequency.
 * This function should be called after watchdog disable.
 */
void cpu_freq_240(void) {
        /* TRM 3.2.3. We must set SEL_0 = 1, SEL_1 = 2 */
        REG(ESP32_RTCCNTL)[28] |= 1UL << 27; /* Register 31.24  SEL0 -> 1 */
        REG(ESP32_DPORT)[15] |= 2UL << 0;    /* Register 5.9    SEL1 -> 2 */
}

/**
 * @brief Configure GPIO pins for UART0 using IO_MUX
 *
 * Configures GPIO1 (TX) and GPIO3 (RX) for UART0 function
 */
static void uart0_gpio_config(void) {
        /* Configure GPIO1 for UART0 TX (function 0) */
        /* Bit fields: [15:12]=function, [11:8]=pullup, [7:4]=pulldown, [3:0]=drive strength */
        REG(IO_MUX_GPIO1_REG)[0] = (0 << 12) | (1 << 8) | (2 << 4);

        /* Configure GPIO3 for UART0 RX (function 0) */
        REG(IO_MUX_GPIO3_REG)[0] = (0 << 12) | (1 << 8) | (2 << 4);
}

/**
 * @brief Get UART base address
 *
 * Returns the base address for the specified UART port.
 *
 * @param port UART port number
 * @return UART base address or NULL if invalid port
 */
static volatile u32* get_uart_base(uart_port_t port) {
        if (port >= UART_PORT_MAX) {
                return NULL;
        }
        return uart_bases[port];
}

/**
 * @brief Read UART register
 *
 * Reads a register from the specified UART port.
 *
 * @param port UART port number
 * @param reg_offset Register offset
 * @return Register value
 */
static u32 uart_read_reg(uart_port_t port, u32 reg_offset) {
        volatile u32* base = get_uart_base(port);
        if (!base) {
                return 0;
        }
        return base[reg_offset / 4];
}

/**
 * @brief Write UART register
 *
 * Writes a value to a register on the specified UART port.
 *
 * @param port UART port number
 * @param reg_offset Register offset
 * @param value Value to write
 */
static void uart_write_reg(uart_port_t port, u32 reg_offset, u32 value) {
        volatile u32* base = get_uart_base(port);
        if (!base) {
                return;
        }
        base[reg_offset / 4] = value;
}

/**
 * @brief Wait for UART TX to be idle
 *
 * Waits until the UART transmit is idle.
 *
 * @param port UART port number
 */
static void uart_wait_tx_idle(uart_port_t port) {
        volatile u32 timeout = 10000; /* Timeout counter */
        while (!(uart_read_reg(port, 0x01C) & UART_STATUS_TX_IDLE) && timeout--) {
                spin(1);
        }
}

/**
 * @brief Initialize UART subsystem
 *
 * Initializes the UART hardware and sets up basic configuration.
 * This function should be called before using any UART functions.
 *
 * @return 0 on success, negative value on error
 */
i32 kio_init(void) {
        /* Initialize all UART ports as disabled */
        for (i32 i = 0; i < UART_PORT_MAX; i++) {
                uart_initialized[i] = false;
        }

        return 0;
}

/**
 * @brief Initialize UART port with default configuration
 *
 * Initializes a UART port with 115200 baud, 8 data bits, 1 stop bit, no parity.
 *
 * @param port UART port number (0, 1, or 2)
 * @return 0 on success, negative value on error
 */
i32 kio_uart_init(uart_port_t port) {
        if (port >= UART_PORT_MAX) {
                return -1;
        }

        /* Configure GPIO pins for UART */
        if (port == UART_PORT_0) {
                uart0_gpio_config();
        }

        /* Reset UART */
        uart_write_reg(port, UART_CONF0_REG, 0);
        spin(100);

        /* Set correct clock divider for 115200 baud */
        uart_write_reg(port, UART_CLKDIV_REG, UART_CLKDIV_115200_80MHZ);

        /* Configure 8N1 (8 data bits, no parity, 1 stop bit) */
        uart_write_reg(port, UART_CONF0_REG, 0x1C);

        /* Enable UART */
        uart_write_reg(port, UART_CONF0_REG, uart_read_reg(port, UART_CONF0_REG) | (1 << 15));

        /* Small delay to ensure UART is properly configured */
        spin(1000);

        /* Mark as initialized */
        uart_initialized[port] = true;

        return 0;
}

/**
 * @brief Send single byte over UART
 *
 * Sends a single byte over the specified UART port.
 * This function blocks until the byte is transmitted.
 *
 * @param port UART port number (0, 1, or 2)
 * @param data Byte to send
 * @return 0 on success, negative value on error
 */
i32 kio_uart_send_byte(uart_port_t port, u8 data) {
        if (port >= UART_PORT_MAX || !uart_initialized[port]) {
                return -1;
        }

        /* Write byte to FIFO */
        uart_write_reg(port, UART_FIFO_REG, data);

        /* Wait for transmission to complete */
        uart_wait_tx_idle(port);

        return 0;
}

/**
 * @brief Send string over UART
 *
 * Sends a null-terminated string over the specified UART port.
 * This function blocks until the entire string is transmitted.
 *
 * @param port UART port number (0, 1, or 2)
 * @param str String to send
 * @return 0 on success, negative value on error
 */
i32 kio_uart_send_string(uart_port_t port, const char* str) {
        if (port >= UART_PORT_MAX || !uart_initialized[port] || !str) {
                return -1;
        }

        while (*str) {
                if (kio_uart_send_byte(port, *str) != 0) {
                        return -1;
                }
                str++;
        }

        return 0;
}

/**
 * @brief Receive single byte from UART
 *
 * Receives a single byte from the specified UART port.
 * This function blocks until a byte is received.
 *
 * @param port UART port number (0, 1, or 2)
 * @param data Pointer to store received byte
 * @return 0 on success, negative value on error
 */
i32 kio_uart_receive_byte(uart_port_t port, u8* data) {
        if (port >= UART_PORT_MAX || !uart_initialized[port] || !data) {
                return -1;
        }

        /* Wait for data to be available in FIFO */
        volatile u32 timeout = 100000; /* Timeout counter */
        while (!(uart_read_reg(port, 0x01C) & 0x0000FF00) && timeout--) {
                spin(1);
        }

        if (timeout == 0) {
                return -1; /* Timeout */
        }

        /* Read byte from FIFO */
        *data = (u8)uart_read_reg(port, UART_FIFO_REG);

        return 0;
}
