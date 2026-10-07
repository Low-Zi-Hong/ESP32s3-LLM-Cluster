#pragma once

#include "esp_log.h"
#include <inttypes.h>

static const char* TAG_VERIFY = "VERIFY_L0";

// 半精度 float16/bfloat16 转单精度 float 辅助函数（若无需精确转换，只看 hex 也可以）
static float fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)(h & 0x7fff) << 13;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;

    if (exp == 0x1f) {
        w = 0x7f800000;
    } else if (exp != 0) {
        w += ((127 - 15) << 23);
    }
    w |= sign;
    float f;
    memcpy(&f, &w, sizeof(float));
    return f;
}

// 打印三值量化权重的前 10 个解包值 {-1, 0, 1}
static void print_ternary_preview(const char* name, const uint8_t* packed_w, float scale, int count = 10) {
    ESP_LOGI(TAG_VERIFY, "--- %s (Scale: %f) ---", name, scale);
    printf("  Packed Raw (前3字节 Hex): 0x%02X 0x%02X 0x%02X\n", packed_w[0], packed_w[1], packed_w[2]);
    printf("  Unpacked Ternary (前%d个): [", count);
    
    int printed = 0;
    for (int byte_idx = 0; printed < count; byte_idx++) {
        uint8_t b = packed_w[byte_idx];
        // 对应 Python 的 2-bit 打包: 1->0b01, -1->0b10, 0->0b00
        for (int shift = 0; shift < 8 && printed < count; shift += 2) {
            uint8_t code = (b >> shift) & 0x03;
            int val = 0;
            if (code == 1) val = 1;
            else if (code == 2) val = -1;
            printf("%d%s", val, (printed == count - 1) ? "" : ", ");
            printed++;
        }
    }
    printf("]\n");
}

// 打印 FP16 浮点数组的前 10 个值
static void print_fp16_preview(const char* name, const uint16_t* weights, int count = 10) {
    printf("  %s (前%d个 FP16): [", name, count);
    for (int i = 0; i < count; i++) {
        printf("%.4f%s", fp16_to_fp32(weights[i]), (i == count - 1) ? "" : ", ");
    }
    printf("]\n");
}

// 专用于打印标准的 32 位浮点数 (float)
static void print_fp32_preview(const char* name, const float* weights, int count = 10) {
    printf("  %s (前%d个 FP32): [", name, count);
    for (int i = 0; i < count; i++) {
        printf("%.5f%s", weights[i], (i == count - 1) ? "" : ", ");
    }
    printf("]\n");
}

void verify_layer_parameters(const TransformerLayer* layer, int layer_idx) {
    ESP_LOGI(TAG_VERIFY, "==================== Layer %d 参数核对 ====================", layer_idx);

    // 重点：必须全部使用 layer-> 访问，绝对不要出现 s_my_layers[0]！
    ESP_LOGI(TAG_VERIFY, "1. rms_norm_1_weight @ %p", layer->rms_norm_1_weight);
    print_fp16_preview("rms_norm_1", layer->rms_norm_1_weight);

    print_ternary_preview("w_q.packed_w", layer->w_q.packed_w, layer->w_q.scale);
    if (layer->w_q.bias) {
        print_fp32_preview("w_q.bias", (const float*)layer->w_q.bias);
    }

    print_ternary_preview("w_k.packed_w", layer->w_k.packed_w, layer->w_k.scale);
    if (layer->w_k.bias) {
        print_fp32_preview("w_k.bias", (const float*)layer->w_k.bias);
    }

    print_ternary_preview("w_v.packed_w", layer->w_v.packed_w, layer->w_v.scale);
    if (layer->w_v.bias) {
        print_fp32_preview("w_v.bias", (const float*)layer->w_v.bias);
    }

    print_ternary_preview("w_o.packed_w", layer->w_o.packed_w, layer->w_o.scale);

    ESP_LOGI(TAG_VERIFY, "6. rms_norm_2_weight @ %p", layer->rms_norm_2_weight);
    print_fp16_preview("rms_norm_2", layer->rms_norm_2_weight);

    print_ternary_preview("w_gate.packed_w", layer->w_gate.packed_w, layer->w_gate.scale);
    print_ternary_preview("w_up.packed_w", layer->w_up.packed_w, layer->w_up.scale);
    print_ternary_preview("w_down.packed_w", layer->w_down.packed_w, layer->w_down.scale);

    ESP_LOGI(TAG_VERIFY, "==========================================================");
}