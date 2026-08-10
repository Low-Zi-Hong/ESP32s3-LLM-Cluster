#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_err.h"
#include "spi_flash_mmap.h"
#include "model_struct.h"
#include "struct.h"

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t v;

    if (exp == 0) {
        if (frac == 0) {
            v = sign << 31;
        } else {
            while (!(frac & 0x400)) {
                frac <<= 1;
                exp -= 1;
            }
            exp += 1;
            frac &= ~0x400;
            v = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 0x1F) {
        v = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
        v = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

struct EmbeddingModule {
    const ModelHeader* header = nullptr;
    const TensorEntry* packed_emb_entry = nullptr;
    const TensorEntry* scales_entry = nullptr;
    const uint8_t* packed_emb_ptr = nullptr;
    const uint16_t* scales_ptr = nullptr;

    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t packed_cols = 0;

    spi_flash_mmap_handle_t mmap_handle;
};

/**
 * @brief 初始化挂载 embed 分区
 * @param partition_label 分区名，默认 "model"
 * @return esp_err_t 挂载状态
 */
esp_err_t embedding_init(EmbeddingModule* emb, const char* partition_label = "model");

/**
 * @brief 根据单个 token_id 查表解包提取 4-bit 向量到 float32 缓冲区中
 * @param token_id 目标词 Token ID
 * @param out_vec 输出向量内存指针 [hidden_size]
 */
void embedding_lookup_token(const EmbeddingModule* emb, int token_id, float* out_vec);

/**
 * @brief 将输入的 Token 序列批量转换为模型矩阵输入 X [Seq_Len, Hidden_Size]
 * @param seq 输入的 TokenID 序列
 * @param out_X 输出矩阵指针 [Seq_Len * Hidden_Size]
 */
void embedding_lookup_sequence(const EmbeddingModule* emb, const TokenSeq* seq, float* out_X);