#pragma once
#include "model_struct.h"

#include <math.h>
#include <string.h>
#include <float.h>

#include "esp_partition.h"
#include "esp_log.h"

esp_err_t init_transformer_layer(TransformerLayer* layers, int num_layers);

void rms_norm(const float* input, const float* weight, float* output, int dim);

void apply_rope(float* vec, int num_heads, int head_dim, int current_pos);

void compute_gpa_attention (
    const float* Q,
    const float* k_cache,
    const float* v_cache,
    int current_pos,
    int max_seq_len,
    float* Attn_out
);

void init_runtime_buffers();

void forward_attention_block(const float* input_X, float* output_X, const TransformerLayer* layer,int layer_idx,int current_pos);