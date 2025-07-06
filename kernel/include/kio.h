#ifndef __KIO_H__
#define __KIO_H__

/**
 * @file kio.h
 * @brief ESP32 UART I/O interface header file
 *
 * Interface for UART communication on the ESP32, including initialization,
 * control, and data transfer functions.
 */

#include "types.h"

/* ESP32 register base addresses - from reference code */
#define ESP32_DPORT       0x3ff00000
#define ESP32_RTCCNTL     0x3ff48000
#define ESP32_TIMERGROUP0 0x3ff5F000
#define ESP32_UART0       0x3ff40000
#define ESP32_UART1       0x3ff50000
#define ESP32_UART2       0x3ff6E000

/* GPIO register addresses - from reference code */
#define GPIO_FUNC_OUT_SEL_CFG_REG 0x3ff44530
#define GPIO_OUT_REG              0x3ff44004
#define GPIO_ENABLE_REG           0x3ff44020
#define GPIO_ENABLE1_REG          0x3ff4402c

/* IO_MUX registers for UART0 */
#define IO_MUX_GPIO1_REG 0x3FF49088 /* UART0 TX */
#define IO_MUX_GPIO3_REG 0x3FF49090 /* UART0 RX */

/* UART register offsets */
#define UART_FIFO_REG   0x000
#define UART_CLKDIV_REG 0x014
#define UART_CONF0_REG  0x020

/* UART configuration bits */
#define UART_CONF0_UART_EN    (1 << 15)
#define UART_CONF0_TX_DIS     (1 << 13)
#define UART_CONF0_RX_DIS     (1 << 14)
#define UART_CONF0_TXFIFO_RST (1 << 11)
#define UART_CONF0_RXFIFO_RST (1 << 12)

/* UART status bits */
#define UART_STATUS_TX_IDLE (1 << 16)

/* Clock divider for 115200 baud at 80MHz UART clock */
/* ESP32 UART uses: divisor = (APB_CLK_FREQ / baud_rate) */
/* For 80MHz APB and 115200 baud: 80000000 / 115200 = 694.4 */
/* Using 694 for closest match */
#define UART_CLKDIV_115200_80MHZ 694

/* UART port enumeration */
typedef enum {
        UART_PORT_0 = 0,
        UART_PORT_1 = 1,
        UART_PORT_2 = 2,
        UART_PORT_MAX
} uart_port_t;

/* Utility macros from reference code */
#define BIT(x) ((uint32_t)1U << (x))
#define REG(x) ((volatile uint32_t*)(x))

/**
 * @brief Disable watchdog timer
 *
 * Disables the ESP32 watchdog timer to prevent system resets.
 * Should be called early in system initialization.
 */
void wdt_disable(void);

/**
 * @brief Set CPU frequency to 240MHz
 *
 * Configures the ESP32 CPU to run at 240MHz frequency.
 * Should be called after watchdog disable.
 */
void cpu_freq_240(void);

/**
 * @brief Initialize UART subsystem
 *
 * Initializes the UART hardware and sets up basic configuration.
 * Should be called before using any UART functions.
 *
 * @return 0 on success, negative value on error
 */
i32 kio_init(void);

/**
 * @brief Initialize UART port with default configuration
 *
 * Initializes a UART port with 115200 baud, 8 data bits, 1 stop bit, no parity.
 *
 * @param port UART port number (0, 1, or 2)
 * @return 0 on success, negative value on error
 */
i32 kio_uart_init(uart_port_t port);

/**
 * @brief Send single byte over UART
 *
 * Sends a single byte over the specified UART port.
 * Blocks until the byte is transmitted.
 *
 * @param port UART port number (0, 1, or 2)
 * @param data Byte to send
 * @return 0 on success, negative value on error
 */
i32 kio_uart_send_byte(uart_port_t port, u8 data);

/**
 * @brief Send string over UART
 *
 * Sends a null-terminated string over the specified UART port.
 * Blocks until the entire string is transmitted.
 *
 * @param port UART port number (0, 1, or 2)
 * @param str String to send
 * @return 0 on success, negative value on error
 */
i32 kio_uart_send_string(uart_port_t port, const char* str);

/**
 * @brief Receive single byte from UART
 *
 * Receives a single byte from the specified UART port.
 * Blocks until a byte is received.
 *
 * @param port UART port number (0, 1, or 2)
 * @param data Pointer to store received byte
 * @return 0 on success, negative value on error
 */
i32 kio_uart_receive_byte(uart_port_t port, u8* data);

#endif /* __KIO_H__ */
