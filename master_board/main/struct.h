#pragma once

#include <stdint.h>

#pragma pack(push, 1)

// Header magic number "ESP3"
struct BinHeader {
    char magic[4];
    uint32_t vocab_size; // vocab size
    uint32_t merge_count;
    uint32_t string_pool_len; // string size
};

struct TokenEntry {
    uint32_t offset; 
    uint16_t length;
    uint16_t reserved;
};

struct MergeRule {
    uint16_t left_id;
    uint16_t right_id;
    uint16_t result_id;
    uint16_t rank;
};

#pragma pack(pop)

#define MAX_TOKENS 256

struct TokenSeq {
    int ids[MAX_TOKENS];
    size_t length = 0;
};