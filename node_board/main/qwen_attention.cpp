#include "qwen_attention.h"

#include "spi_flash_mmap.h"
#include "esp_heap_caps.h"

#define QWEN_HIDDEN_SIZE        896
#define QWEN_INTERMEDIATE_SIZE  4864
#define QWEN_NUM_Q_HEADS        14
#define QWEN_NUM_KV_HEADS       2
#define QWEN_HEAD_DIM           64
#define QWEN_KV_DIM             128
#define QWEN_MLP_DIM            4864

#define QWEN_RMS_NORM_EPS       1e-6f
#define QWEN_ROPE_THERA         1000000.0f

#define NUM_LAYERS_PER_NODE 4

static const char* TAG = "TRANSFORMER";
static spi_flash_mmap_handle_t s_mmap_handle;

#include "esp_timer.h"


#define MAX_CTX_LEN 512
#define NUM_LAYERS_PER_NODE 4
#define QWEN_INTERMEDIATE_SIZE 4864

//kv_cache
static float* s_k_cache[NUM_LAYERS_PER_NODE];
static float* s_v_cache[NUM_LAYERS_PER_NODE];

// temp para
static float* temp_norm_X;
static float* temp_Q;
static float* temp_K;
static float* temp_V;
static float* temp_Attn_Mid;
static float* temp_Mid_X;
static int8_t* temp_X_int8;
static float* temp_Gate;
static float* temp_Up;
static int8_t* temp_MLP_Mid_int8;
static float* temp_MLP_Mid;
static float* temp_Scores;

static float temp_scale;

esp_err_t init_transformer_layer(TransformerLayer* layers, int num_layers) {
    const esp_partition_t* part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40,"model"
    );
    if (!part) return ESP_ERR_NOT_FOUND;

    const void* mmap_base = nullptr;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size,
        ESP_PARTITION_MMAP_DATA, &mmap_base, &s_mmap_handle
    );
    if(err!=ESP_OK) return err;

    ESP_LOGI(TAG, "mmap succesfully! map to: %p", mmap_base);

    const uint8_t* ptr = (const uint8_t*)mmap_base;

    auto advance = [&](size_t bytes) -> const void* {
        const void* current = ptr;
        ptr += bytes;
        return current;
    };

    for (int l = 0; l < num_layers; l++) {
        ESP_LOGI(TAG, "mapping %d 层...", l);

        layers[l].rms_norm_1_weight = (const uint16_t*)advance(QWEN_HIDDEN_SIZE * 2);
        
        layers[l].w_q.in_dim = QWEN_HIDDEN_SIZE;
        layers[l].w_q.out_dim = QWEN_HIDDEN_SIZE;
        layers[l].w_q.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_HIDDEN_SIZE) / 4);
        layers[l].w_q.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_q.bias     = (const float*)advance(QWEN_HIDDEN_SIZE * sizeof(float));

        layers[l].w_k.in_dim = QWEN_HIDDEN_SIZE; 
        layers[l].w_k.out_dim = QWEN_KV_DIM;
        layers[l].w_k.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_KV_DIM) / 4);
        layers[l].w_k.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_k.bias     = (const float*)advance(QWEN_KV_DIM * sizeof(float));

        layers[l].w_v.in_dim = QWEN_HIDDEN_SIZE; 
        layers[l].w_v.out_dim = QWEN_KV_DIM;
        layers[l].w_v.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_KV_DIM) / 4);
        layers[l].w_v.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_v.bias     = (const float*)advance(QWEN_KV_DIM * sizeof(float));

        layers[l].w_o.in_dim = QWEN_HIDDEN_SIZE; 
        layers[l].w_o.out_dim = QWEN_HIDDEN_SIZE;
        layers[l].w_o.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_HIDDEN_SIZE) / 4);
        layers[l].w_o.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_o.bias     = nullptr;

        layers[l].rms_norm_2_weight = (const uint16_t*)advance(QWEN_HIDDEN_SIZE * 2);

        layers[l].w_gate.in_dim = QWEN_HIDDEN_SIZE; 
        layers[l].w_gate.out_dim = QWEN_INTERMEDIATE_SIZE;
        layers[l].w_gate.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_INTERMEDIATE_SIZE) / 4);
        layers[l].w_gate.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_gate.bias     = nullptr;

        layers[l].w_up.in_dim = QWEN_HIDDEN_SIZE; 
        layers[l].w_up.out_dim = QWEN_INTERMEDIATE_SIZE;
        layers[l].w_up.packed_w = (const uint8_t*)advance((QWEN_HIDDEN_SIZE * QWEN_INTERMEDIATE_SIZE) / 4);
        layers[l].w_up.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_up.bias     = nullptr;

        layers[l].w_down.in_dim = QWEN_INTERMEDIATE_SIZE; 
        layers[l].w_down.out_dim = QWEN_HIDDEN_SIZE;
        layers[l].w_down.packed_w = (const uint8_t*)advance((QWEN_INTERMEDIATE_SIZE * QWEN_HIDDEN_SIZE) / 4);
        layers[l].w_down.scale   = *(const float*)advance(sizeof(float));
        layers[l].w_down.bias     = nullptr;
    }

    ESP_LOGI(TAG, "Single layer success: %d Bytes", (int)(ptr - (const uint8_t*)mmap_base));
    return ESP_OK;
}

static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;

    if (exp == 0) {
        // AI 加速：遇到极小非规格化数直接归零 (FTZ)
        v.u = sign;
    } else if (exp == 0x1F) {
        // Inf 或 NaN
        v.u = sign | 0x7F800000 | (mant << 13);
    } else {
        // 常规数值
        v.u = sign | ((exp + 112) << 23) | (mant << 13);
    }
    return v.f;
}

void rms_norm(const float* input, const uint16_t* weight, float* output, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += input[i] * input[i];
    }
    ss /= (float)dim;
    ss += QWEN_RMS_NORM_EPS;
    float inv_rms = 1.0f / sqrtf(ss);

    if (weight != nullptr) {
        for (int i = 0; i < dim; i++){
            // 实时反量化 FP16 的 Gamma 权重
            float w_fp32 = fp16_to_fp32(weight[i]);
            output[i] = input[i] * inv_rms * w_fp32;
        }
    } else {
        for (int i = 0; i < dim; i++){
            output[i] = input[i] * inv_rms;
        }
    }
}

void apply_rope(float* vec, int num_heads, int head_dim, int current_pos) {
    int half_dim = head_dim / 2;
    for(int h = 0; h < num_heads; h++) {
        float* head_vec = vec + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(QWEN_ROPE_THERA, (float)(2*i) / (float)head_dim);
            float val = (float) current_pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            
            float v0 = head_vec[i];
            float v1 = head_vec[i+half_dim];

            head_vec[i] = v0 * cos_val - v1 * sin_val;
            head_vec[i + half_dim] = v0 * sin_val + v1 * cos_val;
        }
    }
}

void compute_gpa_attention (
    const float* Q,
    const float* k_cache,
    const float* v_cache,
    int current_pos,
    int max_seq_len,
    float* Attn_out
) {
    float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);

    int q_heads_per_kv = QWEN_NUM_Q_HEADS / QWEN_NUM_KV_HEADS;

    for (int h = 0; h < QWEN_NUM_Q_HEADS; h++) {
        const float* q_head = Q + h * QWEN_HEAD_DIM;

        int kv_head_idx = h / q_heads_per_kv;

        int kv_head_offset = kv_head_idx * QWEN_HEAD_DIM;

        float* scores = temp_Scores;
        float max_score = -FLT_MAX;

        for (int t = 0; t <= current_pos; t++){
            const float* k_head = k_cache + (t * QWEN_KV_DIM) + kv_head_offset;

            float sum = 0.0f;
            for (int i = 0; i < QWEN_HEAD_DIM; i++) {
                sum+=q_head[i] * k_head[i];
            }

            scores[t] = sum * scale;
            if(scores[t] > max_score) max_score = scores[t];
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t <= current_pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t <= current_pos; t++) {
            scores[t] /= sum_exp;
        }

        float* out_head = Attn_out + h * QWEN_HEAD_DIM;
        memset(out_head,0,QWEN_HEAD_DIM * sizeof(float));

        for (int t = 0; t <= current_pos; t++) {
            const float* v_head = v_cache + (t * QWEN_KV_DIM) + kv_head_offset;
            float weight = scores[t];

            for (int i = 0; i < QWEN_HEAD_DIM;i++){
                out_head[i] += weight * v_head[i];
            }
        }
    }
}

void init_runtime_buffers() {
    for (int i = 0; i < NUM_LAYERS_PER_NODE; i++) {
        s_k_cache[i] = (float*)heap_caps_malloc(MAX_CTX_LEN * QWEN_KV_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
        s_v_cache[i] = (float*)heap_caps_malloc(MAX_CTX_LEN * QWEN_KV_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    }

    temp_norm_X   = (float*)heap_caps_malloc(QWEN_HIDDEN_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_Q        = (float*)heap_caps_malloc(QWEN_HIDDEN_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_K        = (float*)heap_caps_malloc(QWEN_KV_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_V        = (float*)heap_caps_malloc(QWEN_KV_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_Attn_Mid = (float*)heap_caps_malloc(QWEN_HIDDEN_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_Mid_X    = (float*)heap_caps_malloc(QWEN_HIDDEN_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_X_int8   = (int8_t*)heap_caps_malloc(QWEN_HIDDEN_SIZE * sizeof(int8_t), MALLOC_CAP_SPIRAM);
    
    temp_Gate     = (float*)heap_caps_malloc(QWEN_INTERMEDIATE_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_Up       = (float*)heap_caps_malloc(QWEN_INTERMEDIATE_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_MLP_Mid_int8= (int8_t*)heap_caps_malloc(QWEN_INTERMEDIATE_SIZE * sizeof(int8_t), MALLOC_CAP_SPIRAM);
    temp_MLP_Mid    = (float*)heap_caps_malloc(QWEN_INTERMEDIATE_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    temp_Scores   = (float*)heap_caps_malloc(MAX_CTX_LEN * sizeof(float), MALLOC_CAP_SPIRAM);
}

//#define FINE_BENCHMARK

void forward_attention_block_without_q(const float* input_X, float* output_X, const TransformerLayer* layer,int layer_idx,int current_pos){
    rms_norm(input_X, layer->rms_norm_1_weight,temp_norm_X,QWEN_HIDDEN_SIZE);
    
    int64_t t0 = esp_timer_get_time();
    
    bitlinear_forward(temp_norm_X, &layer->w_q, temp_Q);
    bitlinear_forward(temp_norm_X, &layer->w_k, temp_K);
    bitlinear_forward(temp_norm_X, &layer->w_v, temp_V);
    
    int64_t t1 = esp_timer_get_time();

    apply_rope(temp_Q, QWEN_NUM_Q_HEADS, QWEN_HEAD_DIM,current_pos);
    apply_rope(temp_K, QWEN_NUM_KV_HEADS, QWEN_HEAD_DIM,current_pos);

    int64_t t2 = esp_timer_get_time();

    int kv_cache_offset = current_pos * QWEN_KV_DIM;
    memcpy(s_k_cache[layer_idx] + kv_cache_offset, temp_K, QWEN_KV_DIM * sizeof(float));
    memcpy(s_v_cache[layer_idx] + kv_cache_offset, temp_V, QWEN_KV_DIM * sizeof(float));

    int64_t t3 = esp_timer_get_time();

    compute_gpa_attention(temp_Q,s_k_cache[layer_idx],s_v_cache[layer_idx],current_pos,MAX_CTX_LEN,temp_Attn_Mid);

    int64_t t4 = esp_timer_get_time();

    bitlinear_forward(temp_Attn_Mid,&layer->w_o, temp_Q);

    int64_t t5 = esp_timer_get_time();

    for (int i = 0; i < QWEN_HIDDEN_SIZE; i++) {
        temp_Mid_X[i] = input_X[i] + temp_Q[i];
    }

    int64_t t6 = esp_timer_get_time();

    rms_norm(temp_Mid_X, layer->rms_norm_2_weight,temp_norm_X,QWEN_HIDDEN_SIZE);

    int64_t t7 = esp_timer_get_time();

    bitlinear_forward(temp_Mid_X,&layer->w_gate,temp_Gate);
    bitlinear_forward(temp_Mid_X,&layer->w_up,temp_Up);

    int64_t t8 = esp_timer_get_time();

    for (int i = 0; i < QWEN_INTERMEDIATE_SIZE; i++) {
        float x = temp_Gate[i];
        float silu_val = x / (1.0f + expf(-x));
        temp_MLP_Mid[i] = silu_val * temp_Up[i];
    }

    int64_t t9 = esp_timer_get_time();

    bitlinear_forward(temp_MLP_Mid,&layer->w_down,temp_Attn_Mid);

    int64_t t10 = esp_timer_get_time();

    for(int i = 0; i < QWEN_HIDDEN_SIZE; i++) {
        output_X[i] = temp_Mid_X[i] + temp_Attn_Mid[i];
    }
    
    int64_t t11 = esp_timer_get_time();

#ifdef FINE_BENCHMARK
    ESP_LOGI(TAG,"Fine Benchmark: \nQKV mul X: %lld us \n apply rope QK: %lld us \n add kv cache: %lld us\n attention: %lld us \nout_proj:%lld us\n residual connection 1: %lld us \n rms_norm 2: %lld us\n MLP layer: %lld us \n silu:%lld us \n  MLP_down: %lld us \n residual conn 2:%lld us\n", (t1-t0),(t2-t1),(t3-t2),(t4-t3),(t5-t4),(t6-t5),(t7-t6),(t8-t7),(t9-t8),(t10-t9),(t11-t10));
#endif
    
}

void forward_attention_block(const float* input_X, float* output_X, const TransformerLayer* layer,int layer_idx,int current_pos){
    rms_norm(input_X, layer->rms_norm_1_weight,temp_norm_X,QWEN_HIDDEN_SIZE);
    
    int64_t t0 = esp_timer_get_time();

    quantize_X(temp_norm_X,temp_X_int8,QWEN_HIDDEN_SIZE,&temp_scale);
    
    bitlinear_forward_q(temp_X_int8,temp_scale, &layer->w_q, temp_Q);
    bitlinear_forward_q(temp_X_int8,temp_scale, &layer->w_k, temp_K);
    bitlinear_forward_q(temp_X_int8,temp_scale, &layer->w_v, temp_V);
    
    int64_t t1 = esp_timer_get_time();

    apply_rope(temp_Q, QWEN_NUM_Q_HEADS, QWEN_HEAD_DIM,current_pos);
    apply_rope(temp_K, QWEN_NUM_KV_HEADS, QWEN_HEAD_DIM,current_pos);

    int64_t t2 = esp_timer_get_time();

    int kv_cache_offset = current_pos * QWEN_KV_DIM;
    memcpy(s_k_cache[layer_idx] + kv_cache_offset, temp_K, QWEN_KV_DIM * sizeof(float));
    memcpy(s_v_cache[layer_idx] + kv_cache_offset, temp_V, QWEN_KV_DIM * sizeof(float));

    int64_t t3 = esp_timer_get_time();

    compute_gpa_attention(temp_Q,s_k_cache[layer_idx],s_v_cache[layer_idx],current_pos,MAX_CTX_LEN,temp_Attn_Mid);

    int64_t t4 = esp_timer_get_time();

    quantize_X(temp_Attn_Mid, temp_X_int8, QWEN_HIDDEN_SIZE, &temp_scale);
    bitlinear_forward_q(temp_X_int8, temp_scale, &layer->w_o, temp_Q);

    int64_t t5 = esp_timer_get_time();

    for (int i = 0; i < QWEN_HIDDEN_SIZE; i++) {
        temp_Mid_X[i] = input_X[i] + temp_Q[i];
    }

    int64_t t6 = esp_timer_get_time();

    rms_norm(temp_Mid_X, layer->rms_norm_2_weight,temp_norm_X,QWEN_HIDDEN_SIZE);

    int64_t t7 = esp_timer_get_time();

    quantize_X(temp_norm_X,temp_X_int8,QWEN_HIDDEN_SIZE,&temp_scale);

    bitlinear_forward_q(temp_X_int8,temp_scale,&layer->w_gate,temp_Gate);
    bitlinear_forward_q(temp_X_int8,temp_scale,&layer->w_up,temp_Up);

    int64_t t8 = esp_timer_get_time();

    for (int i = 0; i < QWEN_INTERMEDIATE_SIZE; i++) {
        float x = temp_Gate[i];
        float silu_val = x / (1.0f + expf(-x));
        temp_MLP_Mid[i] = silu_val * temp_Up[i];
    }

    quantize_X(temp_MLP_Mid,temp_MLP_Mid_int8,QWEN_INTERMEDIATE_SIZE, &temp_scale);

    int64_t t9 = esp_timer_get_time();

    bitlinear_forward_q(temp_MLP_Mid_int8,temp_scale,&layer->w_down,temp_Attn_Mid);

    int64_t t10 = esp_timer_get_time();

    for(int i = 0; i < QWEN_HIDDEN_SIZE; i++) {
        output_X[i] = temp_Mid_X[i] + temp_Attn_Mid[i];
    }
    
    int64_t t11 = esp_timer_get_time();

#ifdef FINE_BENCHMARK
    ESP_LOGI(TAG,"Fine Benchmark: \nQKV mul X: %lld us \n apply rope QK: %lld us \n add kv cache: %lld us\n attention: %lld us \nout_proj:%lld us\n residual connection 1: %lld us \n rms_norm 2: %lld us\n MLP layer: %lld us \n silu:%lld us \n  MLP_down: %lld us \n residual conn 2:%lld us\n", (t1-t0),(t2-t1),(t3-t2),(t4-t3),(t5-t4),(t6-t5),(t7-t6),(t8-t7),(t9-t8),(t10-t9),(t11-t10));
#endif
    
}