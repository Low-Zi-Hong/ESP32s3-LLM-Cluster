    .text
    .align 4
    .global bitlinear_forward_q_asm
    .type   bitlinear_forward_q_asm, @function

bitlinear_forward_q_asm:
    entry a1, 64        # build a empty area

    # a2: const int8_t* x_vec
    # a3: const uint8_t* packed_w_row
    # a4: uint32_t length

    .literal_position
    movi  a5, LUT_UNPACK_W

    srli    a4, a4, 4   # a4 = length / 4 (loop time)

    ee.zero.accx        # zero out acc

    beqz    a4, .L_dot_end

.L_dot_loop:
    # expand weight

    l8ui    a6, a3, 0    # a6 = Byte 0
    l8ui    a7, a3, 1    # a6 = Byte 1
    l8ui    a8, a3, 2    # a6 = Byte 2
    l8ui    a9, a3, 3    # a6 = Byte 3
    addi    a3, a3, 4    # move pointer back 4 bits

    # LUT table
    slli    a6, a6, 2    # byte offset = byte 0 * 4
    add     a6, a5, a6   # LUT addr + offset
    l32i    a6, a6, 0    # a6 = unpacked 4 int8_t

    slli    a7, a7, 2
    add     a7, a5, a7
    l32i    a7, a7, 0   # a7 = unpacked

    slli    a8, a8, 2
    add     a8, a5, a8
    l32i    a8, a8, 0   # a8 = unpacked

    slli    a9, a9, 2
    add     a9, a5, a9
    l32i    a9, a9, 0   # a9 = unpacked

    # 128-bit vector

    ee.movi.32.q    q1, a6, 0   # fill a6 to q1 channel 0
    ee.movi.32.q    q1, a7, 1   # fill a7 to q1 channel 1
    ee.movi.32.q    q1, a8, 2   # fill a8 to q1 channel 2
    ee.movi.32.q    q1, a9, 3   # fill a9 to q1 channel 3

    ee.vld.128.ip   q0, a2, 16  # load 16 weight and a2 auto +16
    ee.vmulas.s8.accx   q0, q1  # muladd

    addi    a4, a4, -1
    bnez    a4, .L_dot_loop

.L_dot_end:
    # Reduction
    movi     a2, 0
    ee.srs.accx a2, a7, 0
    
    retw
    .size bitlinear_forward_q_asm, . - bitlinear_forward_q_asm
