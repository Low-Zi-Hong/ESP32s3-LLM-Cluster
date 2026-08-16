    .text
    .align 4
    .global bitlinear_forward_asm
    .type   bitlinear_forward_asm, @function

bitlinear_forward_asm:
    entry a1, 64        # build a empty area

    # a2: const float* x_vec
    # a3: const uint8_t* packed_w_row
    # a4: uint32_t length

    ee.vzero.q q7       # empty acc q7
    srli    a4, a4, 2   # a4 = length / 4 (loop time)

    addi    a5, a1, 0   # a5 = &stack_valid_mask
    addi    a6, a1, 16  # a6 = &stack_sign_mask

.L_dot_loop:
    beqz    a4, .L_dot_end  # beqz = Branch if EQual to Zero

    l8ui    a7, a3, 0   # a7 = packed_byte  [Load 8-bit Unsigned Interger]
    addi    a3, a3, 1   # weught pointer ++

    # expand weight

    # 0
    extui   a8, a7, 0, 2    # a8 = get 2-bit (v0) [Extract Unsigned Immediate, extui dest, stc, start_bit, bit_width]
    extui   a9, a8, 1, 1    # a9 = get bit 1 (sign bit)
    slli    a11, a9, 31     # a11 = Sign_mask0 (0x80000000 or 0)    [ Shift Left Logical Immediate]
    extui   a10, a8, 0, 1   # a10 = get bit 0
    or      a10, a10, a9    # a10 = Bit0 | Bit1
    neg     a10, a10        # a10 = Valid_mask0 [ Negate => 0 - src, get inverse sign]
    s32i    a10, a5, 0      # save to stack_valid_mask[0]   [ Store 32-bit integer]
    s32i    a11, a6, 0      # save to stack_sign_mask[0]

    # 1
    extui   a8, a7, 2, 2    # a8 = get 2-bit (v0)
    extui   a9, a8, 1, 1    # a9 = get bit 1 (sign bit)
    slli    a11, a9, 31     # a11 = Sign_mask0 (0x80000000 or 0)
    extui   a10, a8, 0, 1   # a10 = get bit 0
    or      a10, a10, a9    # a10 = Bit0 | Bit1
    neg     a10, a10        # a10 = Valid_mask0
    s32i    a10, a5, 4      # save to stack_valid_mask[1]
    s32i    a11, a6, 4      # save to stack_sign_mask[1]

    # 2
    extui   a8, a7, 4, 2    # a8 = get 2-bit (v0)
    extui   a9, a8, 1, 1    # a9 = get bit 1 (sign bit)
    slli    a11, a9, 31     # a11 = Sign_mask0 (0x80000000 or 0)
    extui   a10, a8, 0, 1   # a10 = get bit 0
    or      a10, a10, a9    # a10 = Bit0 | Bit1
    neg     a10, a10        # a10 = Valid_mask0
    s32i    a10, a5, 8      # save to stack_valid_mask[2]
    s32i    a11, a6, 8      # save to stack_sign_mask[2]

    # 3
    extui   a8, a7, 6, 2    # a8 = get 2-bit (v0)
    extui   a9, a8, 1, 1    # a9 = get bit 1 (sign bit)
    slli    a11, a9, 31     # a11 = Sign_mask0 (0x80000000 or 0)
    extui   a10, a8, 0, 1   # a10 = get bit 0
    or      a10, a10, a9    # a10 = Bit0 | Bit1
    neg     a10, a10        # a10 = Valid_mask0
    s32i    a10, a5, 12     # save to stack_valid_mask[3]
    s32i    a11, a6, 12     # save to stack_sign_mask[3]

    # 128-bit vector

    # 细节：ee.vld.128.ip 会自动把指针 +16，为了不破坏 a5 和 a6 的基地址，用 a12, a13 做中转
    mov     a12, a5         
    mov     a13, a6         

    ee.vld.128.ip   q0, a2, 16      # get 4 float weight    [ Vector Load ] ip-> Increment Pointer
    ee.vld.128.ip   q1, a12, 16     # get 128-bit Valid_mask
    ee.vld.128.ip   q2, a13, 16     # get 128-bit Sign_mask

    ee.vand.q       q3, q0, q1      # q3 = X & Valid_mask
    ee.vxor.q       q3, q3, q2      # q3 = q3 ^ Sign_Mask
    ee.vfadd.s32.q  q7, q7, q3      # acc to q7

    addi   a4, a4, -1      # length -1, countdown how many togo
    j       .L_dot_loop

.L_dot_end:
    # Reduction

    mov a12, a5
    ee.vst.128.ip   q7, a12, 16

    lsi     f0, a5, 0       # load acc[0]   [Load Single-percision Float Immediate]
    lsi     f1, a5, 4       # load acc[1]
    lsi     f2, a5, 8       # load acc[2]
    lsi     f3, a5, 12      # load acc[3]

    add.s   f0, f0, f1      # f0 = acc[0] + acc[1]
    add.s   f2, f2, f3      # f2 = acc[2] + acc[3]
    add.s   f0, f0, f2      # f0 = upper + below
    
    retw
.size bitlinear_forward_asm, . - bitlinear_forward_asm
