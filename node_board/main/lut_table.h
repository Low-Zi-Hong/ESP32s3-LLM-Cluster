#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t data[256];
} UnpackLutType;

extern const UnpackLutType LUT_UNPACK_W;

#ifdef __cplusplus
}
#endif