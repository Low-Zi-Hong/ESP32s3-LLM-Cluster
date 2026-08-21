#include "lm_head.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "LM_HEAD";

int lm_head_sample(const EmbeddingModule* emb_mod, const float* x_final, float temperature){
    uint32_t vocab_size = emb_mod->vocab_size;
    uint32_t hidden_size = emb_mod->hidden_size;
    //const float* weight = emb_mod->weight_table;

    const uint8_t* packed_weight_table = (const uint8_t*)emb_mod->packed_emb_ptr;
    const uint16_t* scales = emb_mod->scales_ptr;

    size_t packed_bytes_pre_row = hidden_size / 2;

    // do matrix mat
    float max_logit = -FLT_MAX;
    int best_token_id = 0;

    // get logits PSRAM around ~128KB
    float* logits = (float*)heap_caps_malloc(vocab_size * sizeof(float),MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!logits) {
        ESP_LOGE(TAG, "PSRAM not enough...");
        return -1;
    }

    for (uint32_t v = 0; v < vocab_size; ++v) {
        const uint8_t* row_packed = packed_weight_table + (v * packed_bytes_pre_row);
        float scale = fp16_to_fp32(scales[v]);

        float acc =0.0f; // accumulator

        uint32_t h = 0;
        for(size_t b = 0; b < packed_bytes_pre_row; ++b) {
            uint8_t val = row_packed[b];

            // unpack first 4 bits
            int8_t w0 = (int8_t)(val & 0x0F);
            if (w0 & 0x08) w0 |= 0xF0;

            // unpack last 4 bits
            int8_t w1 = (int8_t)((val>> 4) & 0x0F);
            if (w1 & 0x08) w1 |= 0xF0;

            acc += x_final[h] * (float)w0;
            acc += x_final[h+1] * (float)w1;

            h+=2;            
        }

        float final_logit = acc * scale;
        logits[v] = final_logit;

        // do Argmax
        if(final_logit > max_logit) {
            max_logit = final_logit;
            best_token_id = (int)v;
        }
    }

    // determine what type of sampling
    if(temperature <= 0.001f) {
        heap_caps_free(logits);
        return best_token_id;
    }

    float sum_exp = 0.0f;
    for (uint32_t v = 0; v < vocab_size; ++v) {
        float exp_val = expf((logits[v] - max_logit) / temperature);
        logits[v] = exp_val;
        sum_exp += exp_val;
    }

    //normalize
    float rand_val = ((float)rand() / (float)RAND_MAX) * sum_exp;
    float cum_sum = 0.0f;
    int sampled_id = best_token_id;

    for (uint32_t v = 0; v<vocab_size;++v){
        cum_sum += logits[v];
        if(cum_sum >= rand_val) {
            sampled_id = (int)v;
            break;
        }
    }

    heap_caps_free(logits);
    return sampled_id;
}