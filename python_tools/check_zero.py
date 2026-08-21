import sys
import os
import numpy as np

def analyze_ternary_sparsity(bin_path, offset_bytes=0):
    if not os.path.exists(bin_path):
        print(f"Error: 找不到文件 {bin_path}")
        return

    file_size = os.path.getsize(bin_path)
    print(f"File: {bin_path}")
    print(f"Total Size: {file_size:,} Bytes ({file_size / (1024*1024):.2f} MB)")

    # 读取二进制文件 (跳过可能的自定义 header 偏移)
    with open(bin_path, "rb") as f:
        if offset_bytes > 0:
            f.seek(offset_bytes)
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    total_bytes = len(raw_bytes)
    total_weights = total_bytes * 4

    print(f"Analyzing {total_weights:,} ternary weights...")

    # 向量化解包 4 个 2-bit 权重
    w0 = (raw_bytes >> 0) & 0x03
    w1 = (raw_bytes >> 2) & 0x03
    w2 = (raw_bytes >> 4) & 0x03
    w3 = (raw_bytes >> 6) & 0x03

    weights = np.concatenate([w0, w1, w2, w3])

    # 统计各类编码数量
    count_zero = np.count_nonzero(weights == 0x00)
    count_pos  = np.count_nonzero(weights == 0x01)
    count_neg  = np.count_nonzero(weights == 0x02)
    count_inv  = np.count_nonzero(weights == 0x03)

    zero_pct = (count_zero / total_weights) * 100
    pos_pct  = (count_pos / total_weights) * 100
    neg_pct  = (count_neg / total_weights) * 100
    inv_pct  = (count_inv / total_weights) * 100

    print("\n" + "=" * 45)
    print("           三值权重分布统计报告            ")
    print("=" * 45)
    print(f"【 0 (Zero)】 : {count_zero:>12,} 个  |  {zero_pct:>6.2f}%")
    print(f"【+1 (Pos) 】 : {count_pos:>12,} 个  |  {pos_pct:>6.2f}%")
    print(f"【-1 (Neg) 】 : {count_neg:>12,} 个  |  {neg_pct:>6.2f}%")
    if count_inv > 0:
        print(f"【0x03 (Invalid)】: {count_inv:>12,} 个  |  {inv_pct:>6.2f}% (未定义编码)")
    print("-" * 45)
    print(f"有效非零参数 (Active) 占比 : {pos_pct + neg_pct:.2f}%")
    print(f"稀疏度 (Sparsity / 含 0 量): {zero_pct:.2f}%")
    print("=" * 45)

    # 硬件优化评估
    if zero_pct >= 50.0:
        print("\n💡 建议：含 0 量 ≥ 50%，采用【跳零/稀疏掩码 (Zero-Skipping)】会有明显的吞吐收益！")
    else:
        print("\n💡 建议：含 0 量 < 50%，采用【无分支展开 (Branchless Matmul)】效率更高，避免分支预测开销。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_sparsity.py <model.bin> [header_offset_bytes]")
    else:
        path = sys.argv[1]
        offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        analyze_ternary_sparsity(path, offset)