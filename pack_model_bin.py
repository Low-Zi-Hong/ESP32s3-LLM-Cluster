import os
import struct
import math
import torch
from safetensors import safe_open

def align_4bytes(buf: bytearray):
    pad_len = (4 - (len(buf) % 4)) % 4
    buf.extend(b'\x00' * pad_len)

def pack_single_bin(tensor_dict, output_bin_path):
    """辅助函数：把传进来的 tensor 字典打包成一个标准对齐的 .bin 文件"""
    tensor_entries = []
    weight_pool = bytearray()
    
    # 保证按名称排序
    keys = sorted(list(tensor_dict.keys()))

    for key in keys:
        tensor = tensor_dict[key]
        
        if tensor.dtype == torch.float16:
            raw_bytes = tensor.numpy().tobytes()
            dtype_code = 0 # FP16
        elif tensor.dtype == torch.float32:
            raw_bytes = tensor.to(torch.float16).numpy().tobytes()
            dtype_code = 0 # FP16
        else: # uint8 / int8 packed
            raw_bytes = tensor.numpy().tobytes()
            dtype_code = 1 # Packed INT8/UINT8

        align_4bytes(weight_pool)
        offset = len(weight_pool)
        length = len(raw_bytes)
        weight_pool.extend(raw_bytes)

        shape = list(tensor.shape) + [1, 1, 1, 1]
        shape = shape[:4]

        tensor_entries.append({
            "name": key,
            "offset": offset,
            "length": length,
            "dtype": dtype_code,
            "shape": shape
        })

    # 构建 Header & Pools
    magic = b'MODL'
    name_pool = bytearray()
    entry_bytes = bytearray()

    for entry in tensor_entries:
        name_bytes = entry["name"].encode('utf-8')
        name_offset = len(name_pool)
        name_len = len(name_bytes)

        name_pool.extend(name_bytes)
        name_pool.append(0)

        entry_bytes.extend(struct.pack(
            '<IHHII4I',
            name_offset, name_len, entry["dtype"],
            entry["offset"], entry["length"],
            entry["shape"][0], entry["shape"][1], entry["shape"][2], entry["shape"][3]
        ))

    header = struct.pack('<4sIIII', magic, len(tensor_entries), len(entry_bytes), len(name_pool), len(weight_pool))

    final_bin = bytearray()
    final_bin.extend(header)
    final_bin.extend(entry_bytes)
    final_bin.extend(name_pool)
    align_4bytes(final_bin)
    final_bin.extend(weight_pool)

    with open(output_bin_path, 'wb') as f:
        f.write(final_bin)

    print(f"  └─ 📦 生成: {os.path.basename(output_bin_path)} | 大小: {len(final_bin) / 1024:.2f} KB ({len(tensor_entries)} 个 Tensor)")

def split_and_pack_safetensors(
    safetensors_path="./cropped_model/model158_bit4.safetensors",
    output_dir="./cropped_model/bins",
    layers_per_bin=3
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始拆分模型: {safetensors_path}")

    all_tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            all_tensors[key] = f.get_tensor(key)

    # 1. 归类 Embedding 相关权重
    embed_tensors = {}
    for k in list(all_tensors.keys()):
        if "embed_tokens" in k or "lm_head" in k:
            embed_tensors[k] = all_tensors.pop(k)
    
    if embed_tensors:
        pack_single_bin(embed_tensors, os.path.join(output_dir, "embed.bin"))

    # 2. 归类 Final Norm 相关权重
    final_norm_tensors = {}
    for k in list(all_tensors.keys()):
        if "model.norm" in k or "final_layernorm" in k:
            final_norm_tensors[k] = all_tensors.pop(k)

    if final_norm_tensors:
        pack_single_bin(final_norm_tensors, os.path.join(output_dir, "final_norm.bin"))

    # 3. 按层（Layer）分组 Transformer Blocks
    # 提取所有出现的 layer index
    layer_indices = set()
    for k in all_tensors.keys():
        if "layers." in k:
            # 提取 "model.layers.0.xxx" 中的数字
            parts = k.split("layers.")
            if len(parts) > 1:
                layer_num = int(parts[1].split(".")[0])
                layer_indices.add(layer_num)

    sorted_layers = sorted(list(layer_indices))
    print(f"检测到总 Layer 层数: {len(sorted_layers)} (层索引: {sorted_layers})")

    # 每 layers_per_bin (3层) 打包成一个 bin
    num_chunks = math.ceil(len(sorted_layers) / layers_per_bin)
    for i in range(num_chunks):
        chunk_layers = sorted_layers[i * layers_per_bin : (i + 1) * layers_per_bin]
        start_l, end_l = chunk_layers[0], chunk_layers[-1]
        
        layer_bin_tensors = {}
        for l in chunk_layers:
            prefix = f"layers.{l}."
            for k in list(all_tensors.keys()):
                if prefix in k:
                    layer_bin_tensors[k] = all_tensors.pop(k)

        bin_name = f"layers_{start_l}_to_{end_l}.bin"
        pack_single_bin(layer_bin_tensors, os.path.join(output_dir, bin_name))

    print("\n✅ 所有文件拆分打包完成！生成的文件列表在:", output_dir)

if __name__ == "__main__":
    split_and_pack_safetensors()