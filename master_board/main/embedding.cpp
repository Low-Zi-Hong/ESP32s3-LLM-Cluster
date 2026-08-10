#include "embedding.h"
#include "struct.h"
#include <string.h>
#include "esp_log.h"
#include "esp_partition.h"

static const char *TAG = "EMBEDDING";

esp_err_t embedding_init(EmbeddingModule* emb, const char* partition_label) {
    if (!emb) return ESP_ERR_INVALID_ARG;

    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, partition_label
    );

    if(!part) {
        ESP_LOGE(TAG, "cannot find partition '%s'", partition_label);
    }

    //map to mmap
    const uint8_t *mmap_base = nullptr;
    esp_err_t err = esp_partition_mmap(
        part,0,part->size,ESP_PARTITION_MMAP_DATA,
        (const void **)&mmap_base,&emb->mmap_handle
    );
    if(err != ESP_OK) {
        ESP_LOGE(TAG, "Embedding fail hangging MMAP: %s", esp_err_to_name(err));
        return err;
    }

    // parse Header
    emb->header = (const ModelHeader*)mmap_base;
    if(memcmp(emb->header->magic, "MODL",4) != 0) {
        ESP_LOGE(TAG,"Embedding magic error");
        return ESP_FAIL;
    }

    const TensorEntry* entries = (const TensorEntry*)(mmap_base + sizeof(ModelHeader));
    const char* name_pool = (const char*)(mmap_base + sizeof(ModelHeader) + emb->header->entry_size);

    uint32_t raw_weight_offset = sizeof(ModelHeader) + emb->header->entry_size + emb->header->name_pool_len;
    uint32_t aligned_weight_offset = (raw_weight_offset + 3) & ~3;
    const uint8_t* weight_pool = mmap_base + aligned_weight_offset;

    for( uint32_t i = 0; i < emb->header->tensor_count; ++i) {
        const char* name = name_pool + entries[i].name_offset;
        if(strstr(name,"weight_packed_4bit") != nullptr) {
            emb->packed_emb_entry = &entries[i];
            emb->packed_emb_ptr = weight_pool + entries[i].offset;
        } else if (strstr(name,"scales")!=nullptr) {
            emb->scales_entry = &entries[i];
            emb->scales_ptr = (const uint16_t*)(weight_pool + entries[i].offset);
        }
    }

    if (!emb->packed_emb_ptr || !emb->scales_ptr){
        ESP_LOGE(TAG, "cannot locate bit 4  weight and scale");
        return ESP_FAIL;
    }

    emb->vocab_size = emb->packed_emb_entry->shape[0];
    emb->packed_cols = emb->packed_emb_entry->shape[1];
    emb->hidden_size = emb->packed_cols * 2;

    ESP_LOGI(TAG, "Embedding succesfully online | vocab: %lu | hidden size: %lu (packed cols: %lu)",
        emb->vocab_size, emb->hidden_size, emb->packed_cols);

    return ESP_OK;
}

void embedding_lookup_token(const EmbeddingModule* emb, int token_id, float* out_vec){
    if(token_id < 0 || (uint32_t)token_id >= emb->vocab_size){
        memset(out_vec, 0, emb->hidden_size * sizeof(float));
        return;
    }

    uint16_t raw_scale = (emb->scales_ptr[token_id]);
    float scale = fp16_to_fp32(raw_scale);

    const uint8_t* row_packed = emb->packed_emb_ptr + ((size_t)token_id * emb->packed_cols);
    
    for (uint32_t j = 0; j<emb->packed_cols; ++j){
        uint8_t packed_val = row_packed[j];

        int8_t w0 = (int8_t)((packed_val >> 4) & 0x0f) - 8;
        int8_t w1 = (int8_t)(packed_val & 0x0f) -8;

        out_vec[j * 2 + 0] = (float)w0 * scale;
        out_vec[j * 2 + 1] = (float)w1 * scale;
    }
}


void embedding_lookup_sequence(const EmbeddingModule* emb, const TokenSeq* seq, float* out_X){
    for(size_t i = 0; i < seq->length; ++i){
        float* current_row = out_X + (i * emb->hidden_size);
        embedding_lookup_token(emb,seq->ids[i],current_row);
    }
}