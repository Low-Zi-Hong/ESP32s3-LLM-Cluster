#pragma once

#include <stdint.h>
#include <stddef.h>
#include <esp_err.h>
#include "embedding.h"

/**
 * @brief 从隐藏特征向量 X 算出所有 Vocabulary 的 Logits，并采样出概率最大的 Token ID
 * 
 * @param emb_mod Embedding 模块指针 (包含 vocab_size, hidden_size 和权重指针)
 * @param x_final 节点反弹回来的最终特征向量，维度 [hidden_size]
 * @param temperature 采样温度 (默认 0.7f，若为 0.0f 则退化为绝对 Argmax 贪婪采样)
 * @return int 最终预测出来的下一个 Token ID
 */
int lm_head_sample(const EmbeddingModule* emb_mod, const float* x_final, float temperature);