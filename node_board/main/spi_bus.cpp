#include "spi_bus.h"
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/spi_master.h"
#include "driver/spi_slave.h"

static const char *TAG = "SPI_BUS";

static spi_device_handle_t s_tx_master_handle = nullptr;

// DRAM special DMA sending and buffer area
static WORD_ALIGNED_ATTR uint8_t s_dma_tx_buf[APP_SPI_MAX_LEN];
static WORD_ALIGNED_ATTR uint8_t s_dma_rx_buf[APP_SPI_MAX_LEN];

esp_err_t spi_bus_init_node(void){

    spi_bus_config_t tx_buscfg = {};
    tx_buscfg.mosi_io_num = SPI_TX_PIN_MOSI;
    tx_buscfg.miso_io_num = -1;
    tx_buscfg.sclk_io_num = SPI_TX_PIN_CLK;
    tx_buscfg.quadwp_io_num = -1;
    tx_buscfg.quadhd_io_num = -1;
    tx_buscfg.max_transfer_sz = APP_SPI_MAX_LEN;

    esp_err_t ret = spi_bus_initialize(SPI2_HOST, &tx_buscfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) return ret;

    spi_device_interface_config_t devcfg = {};
    devcfg.clock_speed_hz = 10 * 1000 * 1000; // 10 MHz
    devcfg.mode = 0;
    devcfg.spics_io_num = SPI_TX_PIN_CS;
    devcfg.queue_size = 7;
    devcfg.cs_ena_pretrans = 2;
    devcfg.cs_ena_posttrans = 2;
    
    ret = spi_bus_add_device(SPI2_HOST, &devcfg, &s_tx_master_handle);
    if(ret!=ESP_OK) return ret;

    // setting out RX
    spi_bus_config_t rx_buscfg = {};
    rx_buscfg.mosi_io_num = SPI_RX_PIN_MOSI;
    rx_buscfg.miso_io_num = -1;
    rx_buscfg.sclk_io_num = SPI_RX_PIN_CLK;
    rx_buscfg.quadwp_io_num = -1;
    rx_buscfg.quadhd_io_num = -1;
    rx_buscfg.max_transfer_sz = APP_SPI_MAX_LEN;

    spi_slave_interface_config_t slvcfg = {};
    slvcfg.mode = 0;
    slvcfg.spics_io_num = SPI_RX_PIN_CS;
    slvcfg.queue_size = 7;
    
    ret = spi_slave_initialize(SPI3_HOST,&rx_buscfg, &slvcfg, SPI_DMA_CH_AUTO);
    if(ret!=ESP_OK) return ret;

    ESP_LOGI(TAG, "dual channel SPI init successfully");
    return ESP_OK;
}

esp_err_t spi_bus_send_frame(SpiPkgType type, const void* payload, size_t payload_len, uint32_t current_pos){
    size_t total_len = sizeof(SpiHeader) + payload_len;
    if (total_len > APP_SPI_MAX_LEN) {
        ESP_LOGE(TAG, "out of range: %u > %d", total_len,APP_SPI_MAX_LEN);
        return ESP_ERR_INVALID_SIZE;
    }

    // build header
    SpiHeader* hdr = (SpiHeader*)s_dma_tx_buf;
    hdr->magic[0] = 'S';
    hdr->magic[1] = 'P';
    hdr->pkg_type = (uint8_t)type;
    hdr->reserved = 0;
    hdr->payload_len = (uint32_t)payload_len;
    hdr->current_pos = current_pos;
    

    // build payload
    if(payload && payload_len > 0){
        memcpy(s_dma_tx_buf + sizeof(SpiHeader),payload,payload_len);
    }

    spi_transaction_t t = {};
    t.length = APP_SPI_MAX_LEN * 8;
    t.tx_buffer = s_dma_tx_buf;

    return spi_device_transmit(s_tx_master_handle, &t);
}

esp_err_t spi_bus_recv_frame(SpiPkgType* out_type, void* out_payload, size_t* out_payload_len,uint32_t* out_current_pos){
    // clear mem
    memset(s_dma_rx_buf, 0, APP_SPI_MAX_LEN);


    // when master read data, need to push a empty frame to run the slave clk
    spi_slave_transaction_t t = {};
    t.length = APP_SPI_MAX_LEN * 8;
    t.rx_buffer = s_dma_rx_buf;

    esp_err_t ret = spi_slave_transmit(SPI3_HOST,&t,portMAX_DELAY);
    if (ret!=ESP_OK) return ret;

    // check magic num
    SpiHeader* hdr = (SpiHeader*)s_dma_rx_buf;
    if(hdr->magic[0] != 'S' || hdr->magic[1] != 'P') {
        //ESP_LOGE(TAG, "SPI 魔数错误! 预期:53 50, 实际收到: %02X %02X", hdr->magic[0], hdr->magic[1]);
        //ESP_LOGE(TAG, "Raw Header Hex: %02X %02X %02X %02X %02X %02X %02X %02X",
        //         s_dma_rx_buf[0], s_dma_rx_buf[1], s_dma_rx_buf[2], s_dma_rx_buf[3],
        //         s_dma_rx_buf[4], s_dma_rx_buf[5], s_dma_rx_buf[6], s_dma_rx_buf[7]);
        return ESP_FAIL;
    }

    if(out_type) *out_type = (SpiPkgType)hdr->pkg_type;
    if(out_payload_len) *out_payload_len = hdr->payload_len;
    if(out_current_pos) *out_current_pos = hdr->current_pos;

    if(out_payload && hdr->payload_len > 0) {
        memcpy(out_payload, s_dma_rx_buf + sizeof(SpiHeader), hdr->payload_len);
    }

    return ESP_OK;
}
