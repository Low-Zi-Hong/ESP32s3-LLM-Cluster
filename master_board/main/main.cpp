/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_system.h"
#include "esp_partition.h"
#include "spi_flash_mmap.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h> // use for memcmp
#include "driver/gpio.h"
#include <esp_err.h>
#include <math.h>

// use psram
#include "esp_heap_caps.h"

// structure data of tokenizer
#include "struct.h"
#include "embedding.h"

//using spi
#include "spi_bus.h"

// lm_head
#include "lm_head.h"

// Global variable
#define RST_OTHER_PIN GPIO_NUM_1
#define OTHER_READY_PIN GPIO_NUM_3

static const char *TAG = "TOKENIZER";
static spi_flash_mmap_handle_t s_token_map_handle;

// global pointer for different area
static const BinHeader* s_header = nullptr;
static const TokenEntry* s_token_table= nullptr;
static const MergeRule* s_merge_table = nullptr;
static const char* s_string_pool = nullptr;
static spi_flash_mmap_handle_t s_mmap_handle;

// byte to id array
static int s_byte_to_id[256];

#define MAX_LINE_LEN 256
//#define MAX_GEN_LEN  128
#define MAX_GEN_LEN  5 //for testing purpose
#define EOS_TOKEN_ID 31997

void init_control_gpio(void) {
    gpio_reset_pin(RST_OTHER_PIN);
    gpio_reset_pin(OTHER_READY_PIN);

    gpio_set_direction(RST_OTHER_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(OTHER_READY_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(OTHER_READY_PIN, GPIO_PULLDOWN_ONLY);

    ESP_LOGI(TAG, "Resetting other esp...");
    gpio_set_level(RST_OTHER_PIN, 1);
    gpio_set_level(RST_OTHER_PIN, 0);
    vTaskDelay(pdMS_TO_TICKS(50));
    gpio_set_level(RST_OTHER_PIN, 1);

    int timeout_ms = 20000; // 给它 2 秒时间变低
    bool node_responded_low = false;

    while (timeout_ms > 0) {
        if (gpio_get_level(OTHER_READY_PIN) == 0) {
            node_responded_low = true;
            break; // 成功捕捉到低电平，顺利过关！
        }
        vTaskDelay(pdMS_TO_TICKS(10)); // 至少 10ms
        timeout_ms -= 10;
    }

if (!node_responded_low) {
        ESP_LOGW(TAG, "警告: Node 没有拉低 READY，可能 Node 正在从极早期 Bootloader 启动中...");
    } else {
        ESP_LOGI(TAG, "捕捉到 Node 已拉低 READY (进入 Boot 流程)...");
    }

    // 3. ⚠️ 关卡二：等待 READY 真正变高 (1) [核心死锁门禁]
    ESP_LOGI(TAG, "等待 Node 加载模型及 SPI DMA 就绪 (等待 READY 变 1)...");
    timeout_ms = 10000; // 10秒超时
    bool node_is_ready = false;

    while (timeout_ms > 0) {
        if (gpio_get_level(OTHER_READY_PIN) == 1) {
            node_is_ready = true;
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(10)); // 释放 CPU 给 IDLE0 自动喂狗！
        timeout_ms -= 10;
    }

    // 🚨 逻辑熔断：如果 Node 没有拉高，必须强行报错并中止，绝对不能打出 READY！
    if (!node_is_ready) {
        ESP_LOGE(TAG, "致命错误: Node 加载超时 (未变高)，强行中止任务！");
        return; // ⛔ 直接 return！绝对不让代码向下执行！
    }

    ESP_LOGI(TAG, "ALL NODE READY!!!");
    
}


int lookup_merge_rule(uint16_t left, uint16_t right) {
    int low = 0;
    int high = s_header-> merge_count -1;
    uint32_t target_key = ((uint32_t)left<<16) | right;

    while (low <= high) {
        int mid = low + (high - low) /2;
        const MergeRule& rule = s_merge_table[mid];
        uint32_t current_key = ((uint32_t)rule.left_id<<16) | rule.right_id;

        if (current_key == target_key) return mid;
        if (current_key < target_key) low = mid +1;
        else high = mid -1;
    }
    return -1;
}

TokenSeq bpe_encode(const char* text) {
    TokenSeq seq = {0};
    const uint8_t* bytes = (const uint8_t*)text;
    size_t text_len = strlen(text);

    // i++ post increment, ++i preincrement
    // i++ return the i +1 and ++i return i but +1 in background
    for (size_t i = 0; i < text_len && seq.length < MAX_TOKENS; ++i) {
        int real_id = s_byte_to_id[bytes[i]];
        seq.ids[seq.length++] = (real_id != -1) ? real_id : 0;
    }

    // bpe sub
    while (seq.length >= 2) {
        int best_merge_idx = -1;
        uint16_t best_rank = 0xFFFF;
        int best_pair_pos = -1;

        for (size_t i = 0; i < seq.length - 1; ++i) {
            int rule_idx = lookup_merge_rule(seq.ids[i], seq.ids[i+1]);
            if (rule_idx != -1 && s_merge_table[rule_idx].rank < best_rank) {
                best_rank = s_merge_table[rule_idx].rank;
                best_merge_idx = rule_idx;
                best_pair_pos = i;
            }
        }
        
        if (best_pair_pos == -1) break; //cannot together again

        seq.ids[best_pair_pos] = s_merge_table[best_merge_idx].result_id;
        
        for (size_t i = best_pair_pos + 1; i < seq.length -1; ++i) {
            seq.ids[i] = seq.ids[i+1];
        }

        seq.length--;

    }
    return seq;
}

size_t seq_to_str(TokenSeq result, char* out_buf, size_t max_buf_len){
    if (!out_buf || max_buf_len == 0) return 0;

    size_t buf_pos = 0;

    for (size_t i = 0; i < result.length; ++i){
        int id = result.ids[i];
        const TokenEntry& entry = s_token_table[id];

        if(buf_pos + entry.length < max_buf_len -1) {
            memcpy(out_buf + buf_pos, s_string_pool + entry.offset, entry.length);
            buf_pos +=entry.length;
        }
        else break;
    }
    out_buf[buf_pos] = '\0';

    return buf_pos;
}

esp_err_t node_pipeline_forward(const float* in_vec, float* out_vec, size_t hidden_size, uint32_t pos) {
    size_t payload_size = hidden_size * sizeof(float);

    esp_err_t err = spi_bus_send_frame(PKG_TYPE_X_MATRIX, (void*)in_vec, payload_size, pos);
    if (err != ESP_OK) return err;

    SpiPkgType rx_type;
    size_t rx_len = 0;
    uint32_t rx_pos = 0;
    return spi_bus_recv_frame(&rx_type,out_vec,&rx_len,&rx_pos);
}

void final_rms_norm_weightless(const float* input, float* output, int size, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    
    // 计算均方根 (RMS)
    float rms = sqrtf((sum_sq / (float)size) + eps);
    float inv_rms = 1.0f / rms;
    
    // 只做缩放，不乘任何 weight 数组
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * inv_rms; 
    }
}

void print_bench_report(int64_t t_prefill_start, int64_t t_prefill_end, int64_t t_decode_start, int64_t t_decode_end, size_t length, int generated_token) {
    // 1. 修复减法变量：必须减去 t_prefill_start
    float prefill_t = (float)(t_prefill_end - t_prefill_start) / 1000000.0f;
    float prefill_tps = (prefill_t > 0.0f) ? ((float)length / prefill_t) : 0.0f;

    // 2. Decode 耗时与 TPS
    float decode_t = (float)(t_decode_end - t_decode_start) / 1000000.0f;
    float decode_tps = (decode_t > 0.0f) ? ((float)generated_token / decode_t) : 0.0f;

    // 3. TTFT (Time To First Token) 建议转换为毫秒 ms 或秒 s
    float ttft_ms = (float)(t_prefill_end - t_prefill_start) / 1000.0f;

    ESP_LOGI("BENCH", 
        "\n================ [BENCHMARK] ================"
        "\n  Prefill Time : %.3f s | Speed: %.2f tok/s (Tokens: %u)"
        "\n  Decode Time  : %.3f s | Speed: %.2f tok/s (Tokens: %d)"
        "\n  TTFT         : %.2f ms"
        "\n=============================================",
        prefill_t, prefill_tps, (unsigned int)length,
        decode_t, decode_tps, generated_token,
        ttft_ms
    );
}

extern "C" void app_main(void)
{
    printf("Hello world!\n");
    init_control_gpio();

    //find partition
    const esp_partition_t *token_part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "token"
    );

    if (!token_part) {
        ESP_LOGE(TAG, "cannot find 'token' partition");
        return;
    }

    // use mmap
    const uint8_t *mmap_base = nullptr;
    ESP_ERROR_CHECK(esp_partition_mmap(
        token_part, 0, token_part->size, ESP_PARTITION_MMAP_DATA,
        (const void **)&mmap_base, &s_mmap_handle
    ));

    s_header = (const BinHeader*)mmap_base;

    if(memcmp(s_header->magic, "ESP3",4) != 0) {
        ESP_LOGE(TAG,"magic num not match");
        return;
    }

    s_token_table = (const TokenEntry*)(mmap_base + sizeof(BinHeader));
    s_merge_table = (const MergeRule*)(mmap_base + sizeof(BinHeader) + (s_header ->vocab_size * sizeof(TokenEntry)));
    s_string_pool = (const char*)(mmap_base + sizeof(BinHeader) + (s_header ->vocab_size * sizeof(TokenEntry)) + (s_header->merge_count * sizeof(MergeRule)));

    // make projection table
    for (int i = 0; i < 256; i++) s_byte_to_id[i] = -1;

    for (uint32_t i = 0; i < s_header->vocab_size; ++i) {
        if(s_token_table[i].length == 1) {
            uint8_t c = (uint8_t)s_string_pool[s_token_table[i].offset];
            if (s_byte_to_id[c] == -1){
                s_byte_to_id[c] = i;
            }
        }
    }


    ESP_LOGI(TAG, "Tokenizer init sucessfull, vocab size: %lu | rule count: %lu",s_header->vocab_size,s_header->merge_count);

    // inti embedding layer
    EmbeddingModule emb_mod;
    if (embedding_init(&emb_mod, "model") != ESP_OK){
        ESP_LOGE("MAIN","embedding fail init");
    }

    ESP_LOGI("MAIN","init SPI master node");

    if (spi_bus_init_node() != ESP_OK) {
        ESP_LOGE("MAIN", "SPI master init fail!");
        return;
    }

    //for real
    char user_input[256];
    float* cur_token_vec = (float*)heap_caps_malloc(emb_mod.hidden_size * sizeof(float),MALLOC_CAP_SPIRAM);
    float* node_out_vec = (float*)heap_caps_malloc(emb_mod.hidden_size * sizeof(float),MALLOC_CAP_SPIRAM);
    float* temp_final_out = (float*)heap_caps_malloc(emb_mod.hidden_size * sizeof(float),MALLOC_CAP_SPIRAM);

    char line[MAX_LINE_LEN];
    int pos = 0;

    printf("\n>>> Qwen 1.58-bit MCU Cluster Ready. Type prompt and press Enter.\n");
    printf("\nUser > ");
    fflush(stdout);

    bool benchmark = false;
    int64_t t_prefill_start;
    int64_t t_prefill_end;
    int64_t t_decode_start;
    int64_t t_decode_end;
    int generated_tokens = 0;
    size_t prompt_token_length = 0;

    while(true) {

        int ch = getchar();
        if (ch != EOF) {

            if (ch == 127 || ch == '\b') {
                if (pos > 0) {
                    pos--;
                    line[pos] = '\0';
                    printf("\b \b");
                    fflush(stdout);
                }
                continue;
            }

            if (ch == '\n' || ch == '\r') {
                if (pos == 0) continue;
                line[pos] = '\0';
                printf("\n");

                if( line[0] == '/'){
                    if(strcmp(line, "/bench") == 0) {
                        benchmark = !benchmark; // 支持开/关切换
                        printf(">>> Benchmark Mode: %s\n", benchmark ? "ON" : "OFF");
                    }
                } else {
                    TokenSeq prompt_token = bpe_encode(line);
                    prompt_token_length = prompt_token.length;
                    if (prompt_token.length > 0) {
                        generated_tokens = 0;
                        printf("Assitant > ");
                        fflush(stdout);

                        t_prefill_start = esp_timer_get_time();
                        uint32_t current_pos = 0;

                        for(size_t i = 0; i < prompt_token.length; i++) {
                            int token_id = prompt_token.ids[i];

                            TokenSeq single_seq = { .ids = token_id, .length = 1};
                            embedding_lookup_sequence(&emb_mod, &single_seq, cur_token_vec);

                            esp_err_t err = node_pipeline_forward(cur_token_vec,node_out_vec,emb_mod.hidden_size,current_pos);
                            if (err!= ESP_OK) {
                                ESP_LOGE(TAG,"pipeline fail...");
                                break;
                            }
                            t_prefill_end = esp_timer_get_time();
                            current_pos++;
                        }

                        t_decode_start = esp_timer_get_time();
                        int next_token_id = -1;

                        for(int step = 0; step < MAX_GEN_LEN; ++step) {
                            final_rms_norm_weightless(node_out_vec, temp_final_out, emb_mod.hidden_size);
                            next_token_id = lm_head_sample(&emb_mod, temp_final_out, 0.0f);

                            if(next_token_id == EOS_TOKEN_ID || next_token_id < 0 || next_token_id >= (int)s_header->vocab_size) {
                                break;
                            }

                            const TokenEntry& entry = s_token_table[next_token_id];
                            generated_tokens++;
                            printf("%.*s", entry.length, s_string_pool + entry.offset);
                            fflush(stdout);

                            TokenSeq next_seq = { .ids = next_token_id, .length = 1};
                            embedding_lookup_sequence(&emb_mod, &next_seq, cur_token_vec);

                            printf("[Step %d] Next ID: %d | Emb[0..3]: %.4f, %.4f, %.4f, %.4f\n", 
                                step, next_token_id, 
                                cur_token_vec[0], cur_token_vec[1], cur_token_vec[2], cur_token_vec[3]);

                            esp_err_t err = node_pipeline_forward(cur_token_vec,node_out_vec,emb_mod.hidden_size,current_pos);
                            if (err != ESP_OK) {
                                ESP_LOGE(TAG, "pipeline forward fail at pos: %lu", current_pos);
                                break;
                            }

                            printf("[Step %d] Out[0..3]: %.4f, %.4f, %.4f, %.4f\n", 
                            step, 
                            node_out_vec[0], node_out_vec[1], node_out_vec[2], node_out_vec[3]);

                            current_pos++;
                        }
                        t_decode_end = esp_timer_get_time();
                    }
                    if(benchmark) print_bench_report(t_prefill_start,t_prefill_end,t_decode_start,t_decode_end,prompt_token_length,generated_tokens);
                    printf("\n\nUser > ");
                }


                pos = 0;
                line[0] = '\0';
                fflush(stdout);
            } else {
                if(pos < MAX_LINE_LEN - 1) {
                    line[pos++] = ch;
                    line[pos] = '\0';
                    putchar(ch);
                    fflush(stdout);
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}