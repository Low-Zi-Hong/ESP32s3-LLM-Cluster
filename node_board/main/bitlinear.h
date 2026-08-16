#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#define QWEN_HIDDEN_SIZE        896
#define QWEN_INTERMEDIATE_SIZE  4864
#define QWEN_NUM_Q_HEADS        14
#define QWEN_NUM_KV_HEADS       2
#define QWEN_HEAD_DIM           64
#define QWEN_KV_DIM             128

#define QWEN_RMS_NORM_EPS       1e-6f
#define QWEN_ROPE_THERA         1000000.0f

// 1.58bit struct
struct LayerWeights {
    const uint8_t* packed_w;
    float scale;
    const float* bias;
    uint32_t in_dim;
    uint32_t out_dim;
};

/**
 * @brief 初始化 Node 逻辑，自动寻找 "model" 分区并完成 MMAP 零拷贝映射
 * @param out_layer 传出初始化好的 LayerWeights 结构体
 */
esp_err_t bitlinear_init(LayerWeights* out_layer);

/**
 * @brief 1.58-bit (Ternary {-1, 0, 1}) 极速矩阵乘法
 * @param input_X  输入特征向量 X
 * @param layer    已加载的层级权重
 * @param output_Y 输出特征向量 Y
 */
void bitlinear_forward(const float* input_X, const LayerWeights* layer, float* output_Y);

#ifdef __cplusplus
extern "C" {
#endif

float bitlinear_forward_asm(const float* x_vec, const uint8_t* packed_w_row, uint32_t length);

#ifdef __cplusplus
}
#endif