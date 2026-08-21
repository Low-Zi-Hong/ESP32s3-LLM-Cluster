#include "bitlinear.h"
#include "esp_partition.h"
#include <cmath>
#include "esp_log.h"
#include "spi_flash_mmap.h"
#include "esp_attr.h"

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

esp_err_t quantize_X(const float* input_X, int8_t* input_X_q, uint32_t dim, float* out_scale){

    if(!input_X || !input_X_q || !out_scale || dim == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    float max_val = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float abs_val = fabsf(input_X[i]);
        if(abs_val > max_val) {
            max_val = abs_val;
        }
    }

    if(max_val < 1e-5f) {
        memset(input_X_q, 0 , dim * sizeof(int8_t));
        *out_scale = 1.0f;
        return ESP_OK;
    }

    float scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;
    *out_scale = scale;

    for (uint32_t i = 0; i < dim; ++i) {
        float val = input_X[i] * inv_scale;

        int32_t q =(int32_t) roundf(val);
        if(q > 127) q = 127;
        if(q<-127) q = -127;

        input_X_q[i] = (int8_t)q;
    }

    return ESP_OK;

}

// raw math
void bitlinear_forward_old(const float* input_X, const LayerWeights* layer, float* output_Y){
    uint32_t packed_bytes_per_row = layer->in_dim /4;

    //later here do divide 2 chunk of row for 2 CPUt
    for (uint32_t row = 0; row < layer->out_dim; ++row){
        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);
        float acc = 0.0f;

        uint32_t x_idx = 0;
        for (uint32_t b = 0; b < packed_bytes_per_row; ++b){
            uint8_t byte_val = w_row[b];

            // real dot!!!
            uint8_t w0 = byte_val & 0x03;
            uint8_t w1 = (byte_val >> 2) & 0x03;
            uint8_t w2 = (byte_val >> 4) & 0x03;
            uint8_t w3 = (byte_val >> 6) & 0x03;

            if(w0 == 0x01) acc += input_X[x_idx];
            else if(w0 == 0x02) acc -= input_X[x_idx];
            else if(w0 == 0x03) ESP_LOGE(TAG, "Catch ya!");
            x_idx++;

            if(w1 == 0x01) acc += input_X[x_idx];
            else if(w1 == 0x02) acc -= input_X[x_idx];
            else if(w0 == 0x03) ESP_LOGE(TAG, "Catch ya!");
            x_idx++;
            
            if(w2 == 0x01) acc += input_X[x_idx];
            else if(w2 == 0x02) acc -= input_X[x_idx];
            x_idx++;
            
            if(w3 == 0x01) acc += input_X[x_idx];
            else if(w3 == 0x02) acc -= input_X[x_idx];
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

void bitlinear_forward_one_cpu_asm(const float* input_X, const LayerWeights* layer, float* output_Y){
    uint32_t packed_bytes_per_row = layer->in_dim / 4;

    for (uint32_t row = 0; row < layer->out_dim; ++row) {

        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);

        float acc = bitlinear_forward_asm(input_X, w_row, layer->in_dim);

        float final_val = acc * layer->scale;
        if (layer->bias != nullptr) {
            final_val += layer->bias[row];
        }

        output_Y[row] = final_val;

    }
}

void bitlinear_worker_task(void* arg){
    BitLinearTaskArgs* args = (BitLinearTaskArgs*)arg;
    uint32_t packed_bytes_per_row = args->layer->in_dim /4;
    
    for (uint32_t row = args->start_row; row < args->end_row; ++row) {
        const uint8_t* w_row = args->layer->packed_w + (row * packed_bytes_per_row);

        float acc = bitlinear_forward_asm(args->input_X,w_row,args->layer->in_dim);

        float final_val = acc * args->layer->scale;
        if(args->layer->bias != nullptr) {
            final_val += args->layer->bias[row];
        }

        args->output_Y[row] = final_val;
    }

    xSemaphoreGive(args->done_sem);
    vTaskDelete(NULL);

}


void bitlinear_forward(const float* input_X, const LayerWeights* layer, float* output_Y){
    uint32_t half_row = layer->out_dim /2;
    uint32_t packed_bytes_per_row = layer->in_dim /4;

    SemaphoreHandle_t done_sem = xSemaphoreCreateBinary();

    BitLinearTaskArgs worker_args = {
        input_X, layer, output_Y,
        0, half_row, done_sem
    };

    BaseType_t current_core = xPortGetCoreID();
    BaseType_t target_core = (current_core == 0) ? 1 : 0;

    xTaskCreatePinnedToCore(
        bitlinear_worker_task,
        "bitlinear_worker",
        4096,
        &worker_args,
        configMAX_PRIORITIES - 1,
        NULL,
        target_core
    );

    for (uint32_t row = half_row; row < layer->out_dim; ++row){
        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);

        float acc = bitlinear_forward_asm(input_X,w_row,layer->in_dim);

        float final_val = acc * layer->scale;
        if(layer->bias != nullptr) {
            final_val += layer->bias[row];
        }

        output_Y[row] = final_val;
    }

    xSemaphoreTake(done_sem, portMAX_DELAY);

    vSemaphoreDelete(done_sem);


}

void bitlinear_forward_q_cpp(const int8_t* input_X_q,const float scale, const LayerWeights* layer, float* output_Y){
    uint32_t packed_bytes_per_row = layer->in_dim /4;
    float f_scale = scale * layer->scale;

    //later here do divide 2 chunk of row for 2 CPUt
    for (uint32_t row = 0; row < layer->out_dim; ++row){
        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);
        int32_t acc = 0;

        uint32_t x_idx = 0;
        for (uint32_t b = 0; b < packed_bytes_per_row; ++b){
            uint8_t byte_val = w_row[b];

            // real dot!!!
            uint8_t w0 = byte_val & 0x03;
            uint8_t w1 = (byte_val >> 2) & 0x03;
            uint8_t w2 = (byte_val >> 4) & 0x03;
            uint8_t w3 = (byte_val >> 6) & 0x03;

            if(w0 == 0x01) acc += input_X_q[x_idx];
            else if(w0 == 0x02) acc -= input_X_q[x_idx];
            else if(w0 == 0x03) ESP_LOGE(TAG, "Catch ya!");
            x_idx++;

            if(w1 == 0x01) acc += input_X_q[x_idx];
            else if(w1 == 0x02) acc -= input_X_q[x_idx];
            else if(w0 == 0x03) ESP_LOGE(TAG, "Catch ya!");
            x_idx++;
            
            if(w2 == 0x01) acc += input_X_q[x_idx];
            else if(w2 == 0x02) acc -= input_X_q[x_idx];
            x_idx++;
            
            if(w3 == 0x01) acc += input_X_q[x_idx];
            else if(w3 == 0x02) acc -= input_X_q[x_idx];
            x_idx++;
        }

        float final_val = (float)acc * f_scale;

        if (layer->bias != nullptr) {
            final_val += layer->bias[row];
        }

        output_Y[row] = final_val;
    }
}

void bitlinear_worker_task_q(void* arg){
    BitLinearTaskArgsQ* args = (BitLinearTaskArgsQ*)arg;
    uint32_t packed_bytes_per_row = args->layer->in_dim /4;
    
    for (uint32_t row = args->start_row; row < args->end_row; ++row) {
        const uint8_t* w_row = args->layer->packed_w + (row * packed_bytes_per_row);

        float acc = bitlinear_forward_q_asm(args->input_X_q,w_row,args->layer->in_dim);

        float final_val = acc * args->scale;
        if(args->layer->bias != nullptr) {
            final_val += args->layer->bias[row];
        }

        args->output_Y[row] = final_val;
    }

    xSemaphoreGive(args->done_sem);
    vTaskDelete(NULL);

}

void bitlinear_forward_q(const int8_t* input_X_q,const float scale,  const LayerWeights* layer, float* output_Y){
    uint32_t half_row = layer->out_dim /2;
    uint32_t packed_bytes_per_row = layer->in_dim /4;
    float f_scale = scale * layer->scale;

    SemaphoreHandle_t done_sem = xSemaphoreCreateBinary();

    BitLinearTaskArgsQ worker_args = {
        input_X_q,f_scale, layer, output_Y,
        0, half_row, done_sem
    };

    BaseType_t current_core = xPortGetCoreID();
    BaseType_t target_core = (current_core == 0) ? 1 : 0;

    xTaskCreatePinnedToCore(
        bitlinear_worker_task_q,
        "bitlinear_worker",
        4096,
        &worker_args,
        configMAX_PRIORITIES - 1,
        NULL,
        target_core
    );

    for (uint32_t row = half_row; row < layer->out_dim; ++row){
        const uint8_t* w_row = layer->packed_w + (row * packed_bytes_per_row);

        int32_t acc = bitlinear_forward_q_asm(input_X_q,w_row,layer->in_dim);

        float final_val = (float)acc * f_scale;
        if(layer->bias != nullptr) {
            final_val += layer->bias[row];
        }

        output_Y[row] = final_val;
    }

    xSemaphoreTake(done_sem, portMAX_DELAY);

    vSemaphoreDelete(done_sem);


}