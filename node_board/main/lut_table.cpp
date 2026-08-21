#include "lut_table.h"

constexpr uint8_t map_2bit_to_int8(uint8_t val_2bit) {
    // 00-> 0x00
    // 01 -> 0x01
    // 10 -> 0xFF
    // 11 -> return err 0x00
    return (val_2bit == 1) ? 0x01:((val_2bit == 2)? 0xFF : 0x00);
}

constexpr uint32_t unpack_byte_to_uint32(uint8_t byte_val) {
    uint8_t w0 = map_2bit_to_int8((byte_val>>0) & 0x03);
    uint8_t w1 = map_2bit_to_int8((byte_val>>2) & 0x03);
    uint8_t w2 = map_2bit_to_int8((byte_val>>4) & 0x03);
    uint8_t w3 = map_2bit_to_int8((byte_val>>6) & 0x03);

    return ((uint32_t)w0) |
        ((uint32_t)w1 << 8) |
        ((uint32_t)w2 << 16) |
        ((uint32_t)w3 << 24);

}

struct LutGenerator {
    UnpackLutType lut;

    constexpr LutGenerator() : lut{} {
        for(int i = 0; i < 256; ++i){
            lut.data[i] = unpack_byte_to_uint32((uint8_t)i);
        }
    }
};

alignas(16) static constexpr LutGenerator s_gen{};

extern "C" alignas(16) const UnpackLutType LUT_UNPACK_W = s_gen.lut; //cpp auto save here