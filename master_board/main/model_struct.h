#pragma once
#include <stdint.h>

#pragma pack(push,1)

struct ModelHeader
{
    char magic[4];
    uint32_t tensor_count;
    uint32_t entry_size;
    uint32_t name_pool_len;
    uint32_t weight_size;
};

struct TensorEntry
{
    uint32_t name_offset;
    uint16_t name_len;
    uint16_t dtype;
    uint32_t offset;
    uint32_t length;
    uint32_t shape[4];
};

#pragma pack(pop)