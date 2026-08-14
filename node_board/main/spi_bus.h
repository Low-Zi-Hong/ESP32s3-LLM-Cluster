#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

// channel A: TX sending  mode
#define SPI_TX_PIN_CS   4
#define SPI_TX_PIN_MOSI 5
#define SPI_TX_PIN_CLK  7

// channel B: RX rev mode
#define SPI_RX_PIN_CS   15
#define SPI_RX_PIN_MOSI   16
#define SPI_RX_PIN_CLK   18

#define APP_SPI_MAX_LEN 4096

// define struct
enum SpiPkgType : uint8_t {
    PKG_TYPE_X_MATRIX = 0x01,
    PKG_TYPE_ACK = 0x02,
};

// 8bits header
struct __attribute__((packed)) SpiHeader {
    uint8_t magic[2];
    uint8_t pkg_type;
    uint8_t reserved;
    uint32_t payload_len;
    uint32_t current_pos;
};

/**
 * @brief 初始化 SPI 总线
 * @param is_master true: Master 主板, false: Node 从板
 */
esp_err_t spi_bus_init_node(void);

/**
 * @brief 通用带 Header 发送函数（自动打包魔数与长度）
 */
esp_err_t spi_bus_send_frame(SpiPkgType type, const void* payload, size_t payload_len, uint32_t current_pos);

/**
 * @brief 通用带 Header 接收函数（自动校验魔数与解析长度）
 */
esp_err_t spi_bus_recv_frame(SpiPkgType* out_type, void* out_payload, size_t* out_payload_len,uint32_t* out_current_pos);
