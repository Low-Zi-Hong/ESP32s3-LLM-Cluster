#include "bitlinear.h"
#include "esp_partition.h"
#include <cmath>
#include "esp_log.h"
#include "spi_flash_mmap.h"

static const char* TAG = "BITLINEAR";
static spi_flash_mmap_handle_t s_mmap_handle;

esp_err_t bitlinear_init(LayerWeights* out_layer){
    const esp_partition_t* model_part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model"
    );

    if(!model_part) {
        ESP_LOGE(TAG, "cannot find 'model' partition");
        return ESP_ERR_NOT_FOUND;
    }

    const void* mmap_base = nullptr;
    esp_err_t err = esp_partition_mmap(
        model_part, 0 , model_part->size,
        ESP_PARTITION_MMAP_DATA, &mmap_base, &s_mmap_handle
    );

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Flash mmap fail!");
        return err;
    }

    // setting node
    out_layer->in_dim = 896;
    out_layer->out_dim = 896;


    //not sure the achitecture of the bin file... need to refer back the py file but just gambling lol
    uint32_t packed_weight_size = (out_layer->in_dim * out_layer->out_dim) / 4;

    out_layer->packed_w = (const uint8_t*)mmap_base;
    out_layer->scale = *(const float*)((const uint8_t*)mmap_base + packed_weight_size);

    ESP_LOGI(TAG, "1.58-bit weight mmap successfully! size: %lu bytes", packed_weight_size);
    return ESP_OK;
}

void bitlinear_forward(const float* input_X, const LayerWeights* layer, float* output_Y){
    uint32_t packed_bytes_per_row = layer->in_dim /4;

    for (uint32_t row = 0; row < layer->out_dim; ++row){
        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);
        float acc = 0.0f;

        uint32_t x_idx = 0;
        for (uint32_t b = 0; b < packed_bytes_per_row; ++b){
            uint8_t byte_val = w_row[b];

            // real dot!!!
            uint8_t w0 = byte_val & 0x03;
            if(w0 == 0x01) acc += input_X[x_idx];
            else if(w0 == 0x03) acc -= input_X[x_idx];
            x_idx++;

            uint8_t w1 = (byte_val >> 2) & 0x03;
            if(w1 == 0x01) acc += input_X[x_idx];
            else if(w1 == 0x03) acc -= input_X[x_idx];
            x_idx++;
            
            uint8_t w2 = (byte_val >> 4) & 0x03;
            if(w2 == 0x01) acc += input_X[x_idx];
            else if(w2 == 0x03) acc -= input_X[x_idx];
            x_idx++;
            
            uint8_t w3 = (byte_val >> 6) & 0x03;
            if(w3 == 0x01) acc += input_X[x_idx];
            else if(w3 == 0x03) acc -= input_X[x_idx];
            x_idx++;
        }

        float final_val = acc * layer->scale;

        if (layer->bias != nullptr) {
            final_val += layer->bias[row];
        }

        if (isnan(final_val) || isinf(final_val)) {
            ESP_LOGE("BITLINEAR", "💥 抓到 NaN! row=%lu, acc=%f, scale=%f", row, acc, layer->scale);
            // 把这一行的前几个输入打出来看看是不是已经坏了
            ESP_LOGE("BITLINEAR", "   input_X[0]=%f, input_X[1]=%f", input_X[0], input_X[1]);
        }

        output_Y[row] = final_val;
    }
}