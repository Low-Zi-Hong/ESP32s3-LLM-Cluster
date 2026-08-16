#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "spi_bus.h" // 确保引用的是双通道脚本
#include "bitlinear.h"
#include "model_struct.h"
#include "qwen_attention.h"
#include "driver/gpio.h"
#include "rom/ets_sys.h"

#define NUM_LAYERS_PER_NODE 4

static const char* TAG = "NODE_MAIN";

static TransformerLayer s_my_layers[NUM_LAYERS_PER_NODE];

#define WORK_LED_GPIO GPIO_NUM_8
#define RGB_LED_GPIO GPIO_NUM_48 // 钛合金狗眼

// rst other esp
#define RST_OTHER_PIN GPIO_NUM_1 
#define READY_PIN GPIO_NUM_3

void init_control_gpio(void) {
    gpio_reset_pin(RST_OTHER_PIN);
    gpio_reset_pin(READY_PIN);

    gpio_set_direction(RST_OTHER_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(READY_PIN, GPIO_MODE_OUTPUT);

    gpio_set_level(READY_PIN, 0);


    gpio_set_level(RST_OTHER_PIN, 1);
    gpio_set_level(RST_OTHER_PIN, 0);
    vTaskDelay(pdMS_TO_TICKS(50));
    gpio_set_level(RST_OTHER_PIN, 1);

}

void init_work_led() {
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;           
    io_conf.mode = GPIO_MODE_OUTPUT;                 
    io_conf.pin_bit_mask = (1ULL << WORK_LED_GPIO);  
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;    
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;        
    gpio_config(&io_conf);

    gpio_set_level(WORK_LED_GPIO, 0);
}

void kill_board_rgb() {
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << RGB_LED_GPIO);
    io_conf.pull_down_en = GPIO_PULLDOWN_ENABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);

    // 1. 先死死拉低引脚 500 微秒，强制重置 WS2812B 芯片状态
    gpio_set_level(RGB_LED_GPIO, 0);
    ets_delay_us(500);

    // 2. 模拟发送 24 个 0 位 (简单的低电平脉冲) 强行清空 RGB 寄存器
    for (int i = 0; i < 24; i++) {
        gpio_set_level(RGB_LED_GPIO, 0);
        ets_delay_us(50);
    }
}

extern "C" void app_main(void)
{   
    init_work_led();
    kill_board_rgb(); // 关掉钛合金狗眼

    float* buf_A = (float*)heap_caps_aligned_alloc(16,896 * sizeof(float), MALLOC_CAP_SPIRAM);
    float* buf_B = (float*)heap_caps_aligned_alloc(16,896 * sizeof(float), MALLOC_CAP_SPIRAM);

    float* s_node_input_X = (float*)heap_caps_malloc(
        896 * sizeof(float), 
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    float* s_node_out_Y = (float*)heap_caps_malloc(
        896 * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    ESP_LOGI(TAG, "int layers");
    if (init_transformer_layer(s_my_layers,NUM_LAYERS_PER_NODE) != ESP_OK) {
        ESP_LOGE(TAG, "cannot load model!");
        return;
    }
    init_runtime_buffers();

    if (spi_bus_init_node() != ESP_OK) return;
    
    init_control_gpio();
    gpio_set_level(READY_PIN, 1); // tell back master I am ready

    while (true) {
        SpiPkgType type;
        uint32_t raw_current_pos;
        size_t rx_len = 0;

        // 1. 在 RX 通道阻塞等待 Master 把 Token 矩阵压过来
        esp_err_t err = spi_bus_recv_frame(&type, buf_A, &rx_len, &raw_current_pos);

        if (err == ESP_OK) {
            ESP_LOGI(TAG, ">>> 收到 Master 压入的数据包: %u 字节", rx_len);
            gpio_set_level(WORK_LED_GPIO, 1);

            int64_t t0 = esp_timer_get_time();

            float* current_in = buf_A;
            float* current_out = buf_B;

            int current_pos = (int)raw_current_pos;

            for (int l = 0; l < NUM_LAYERS_PER_NODE; l++) {
                forward_attention_block(current_in, current_out, &s_my_layers[l],l,current_pos);

                float* temp = current_in;
                current_in = current_out;
                current_out = temp;
            }

            int64_t t1 = esp_timer_get_time();
            ESP_LOGI(TAG, "BitLinear 计算耗时: %lld us", (t1 - t0));

            // 2. 算完后，通过 TX 通道主动推回给 Master
            ESP_LOGI(TAG, "正在将结果推回给 Master...");
            spi_bus_send_frame(PKG_TYPE_X_MATRIX, current_in, rx_len,raw_current_pos);
            gpio_set_level(WORK_LED_GPIO, 0);
        }
    }
}