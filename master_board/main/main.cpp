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

// use psram
#include "esp_heap_caps.h"

// structure data of tokenizer
#include "struct.h"
#include "embedding.h"

// Global variable

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
    TokenSeq seq;
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

extern "C" void app_main(void)
{
    printf("Hello world!\n");

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

     // store X in PSRAM
    float* s_matrix_X = (float*)heap_caps_malloc(
        MAX_TOKENS * emb_mod.hidden_size * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!s_matrix_X) {
        ESP_LOGE("MAIN", "PSRAM out of memory, fail allocate X");
        return;
    }

    //test
    const char* test_text = "hello马来西亚";
    ESP_LOGI(TAG, "test text: '%s'",test_text);

    int64_t t0 = esp_timer_get_time();
    TokenSeq result = bpe_encode(test_text);
    int64_t t1 = esp_timer_get_time();

    ESP_LOGI(TAG, "time spent: %lld us",(t1-t0));
    ESP_LOGI(TAG, "result:");
    for (size_t i = 0; i < result.length; ++i) {
        int id = result.ids[i];
        const TokenEntry& entry = s_token_table[id];
        printf("%d(%.*s) ", id, entry.length, s_string_pool + entry.offset);
    }
    printf("\n");

    // output real string
    char decode_buf[512] = {0};
    seq_to_str(result,decode_buf,sizeof(decode_buf));

    ESP_LOGI(TAG, "real result: '%s'",decode_buf);

    // 4. 将 TokenSeq 查表合成为 Transformer 输入矩阵 X
    int64_t t3 = esp_timer_get_time();
    embedding_lookup_sequence(&emb_mod, &result, s_matrix_X);
    int64_t t4 = esp_timer_get_time();

    ESP_LOGI("MAIN", "Embedding 查表解包完成，耗时: %lld us", (t4 - t3));
    ESP_LOGI("MAIN", "生成矩阵 X 维度: [%d, %lu]", result.length, emb_mod.hidden_size);

    // 打印第一个 Token 对应向量的前 5 个元素作为验证
    printf("Token 0 ('%d') 向量前 5 维数据: ", result.ids[0]);
    for (int k = 0; k < 5; ++k) {
        printf("%.4f ", s_matrix_X[k]);
    }
    printf("\n");


}