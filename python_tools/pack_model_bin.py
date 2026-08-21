import os
import struct
import math
import torch
from safetensors import safe_open

def align_4bytes(buf: bytearray):
    pad_len = (4 - (len(buf) % 4)) % 4
    buf.extend(b'\x00' * pad_len)

# =================================================================
# 保留给 Master 用的原始打包格式 (带 Header，处理 int4 / bf16)
# =================================================================
def pack_single_bin(tensor_dict, output_bin_path):
    tensor_entries = []
    weight_pool = bytearray()
    
    keys = sorted(list(tensor_dict.keys()))

    for key in keys:
        tensor = tensor_dict[key]
        if tensor.dtype == torch.float16:
            raw_bytes = tensor.numpy().tobytes()
            dtype_code = 0
        elif tensor.dtype == torch.float32:
            raw_bytes = tensor.to(torch.float16).numpy().tobytes()
            dtype_code = 0
        else:
            raw_bytes = tensor.numpy().tobytes()
            dtype_code = 1

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

def export_raw_layer_bin(layer_tensors, layer_idx):
    final_bin = bytearray()
    
    def append_tensor(key_suffix, target_dtype=None, default_shape=None, default_val=0.0):
        matched_key = None
        for k in layer_tensors.keys():
            if k.endswith(key_suffix):
                matched_key = k
                break
        
        if not matched_key:
            # ⚠️ 致命修复：如果找不到权重，必须严格按照指定的类型和尺寸填入占位符字节！
            # 绝对不能直接 return，否则 C++ 指针会读进异次元！
            if target_dtype is not None and default_shape is not None:
                pad_tensor = torch.full((default_shape,), default_val, dtype=target_dtype)
                final_bin.extend(pad_tensor.numpy().tobytes())
            return

        tensor = layer_tensors[matched_key]
        
        if target_dtype is not None:
            # 严格按照 C++ 的胃口，强转为 FP16 或 FP32
            tensor = tensor.to(target_dtype)
        
        final_bin.extend(tensor.numpy().tobytes())

    # =================================================================
    # 严格按照 C++ advance() 的字节数写入！错一个 byte 都会死锁！
    # =================================================================
    
    # 1. Attention 前的 RMSNorm (C++ advance: 896 * 2) -> 必须是 FP16!
    append_tensor("input_layernorm.weight", target_dtype=torch.float16, default_shape=896, default_val=1.0)
    
    # 2. Q proj
    append_tensor("self_attn.q_proj.weight_packed", target_dtype=None) 
    append_tensor("self_attn.q_proj.gamma", target_dtype=torch.float32) # C++ sizeof(float)
    append_tensor("self_attn.q_proj.bias", target_dtype=torch.float32, default_shape=896, default_val=0.0)
    
    # 3. K proj (GQA 维度 128)
    append_tensor("self_attn.k_proj.weight_packed", target_dtype=None)
    append_tensor("self_attn.k_proj.gamma", target_dtype=torch.float32)
    append_tensor("self_attn.k_proj.bias", target_dtype=torch.float32, default_shape=128, default_val=0.0)
    
    # 4. V proj (GQA 维度 128)
    append_tensor("self_attn.v_proj.weight_packed", target_dtype=None)
    append_tensor("self_attn.v_proj.gamma", target_dtype=torch.float32)
    append_tensor("self_attn.v_proj.bias", target_dtype=torch.float32, default_shape=128, default_val=0.0)
    
    # 5. O proj
    append_tensor("self_attn.o_proj.weight_packed", target_dtype=None)
    append_tensor("self_attn.o_proj.gamma", target_dtype=torch.float32)
    # C++ 里 O_proj bias 是 nullptr，所以这里不写 bias!
    
    # 6. MLP 前的 RMSNorm (C++ advance: 896 * 2) -> 必须是 FP16!
    append_tensor("post_attention_layernorm.weight", target_dtype=torch.float16, default_shape=896, default_val=1.0)
    
    # 7. Gate proj
    append_tensor("mlp.gate_proj.weight_packed", target_dtype=None)
    append_tensor("mlp.gate_proj.gamma", target_dtype=torch.float32)
    
    # 8. Up proj
    append_tensor("mlp.up_proj.weight_packed", target_dtype=None)
    append_tensor("mlp.up_proj.gamma", target_dtype=torch.float32)
    
    # 9. Down proj
    append_tensor("mlp.down_proj.weight_packed", target_dtype=None)
    append_tensor("mlp.down_proj.gamma", target_dtype=torch.float32)

    return final_bin
    
def split_and_pack_safetensors(
    safetensors_path="../cropped_Qwen/qwen_158_int4.safetensors",
    output_dir="../cropped_Qwen/bins",
    layers_per_bin=4
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始拆分模型 (Node 极致模式): {safetensors_path}")

    all_tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            all_tensors[key] = f.get_tensor(key)

    # 1. 归类 Embedding 相关权重 (保持原样给 Master)
    embed_tensors = {}
    for k in list(all_tensors.keys()):
        if "embed_tokens" in k or "lm_head" in k:
            embed_tensors[k] = all_tensors.pop(k)
    if embed_tensors:
        pack_single_bin(embed_tensors, os.path.join(output_dir, "embed.bin"))

    # 2. 归类 Final Norm 相关权重 (保持原样给 Master)
    final_norm_tensors = {}
    for k in list(all_tensors.keys()):
        if "model.norm" in k or "final_layernorm" in k:
            final_norm_tensors[k] = all_tensors.pop(k)
    if final_norm_tensors:
        pack_single_bin(final_norm_tensors, os.path.join(output_dir, "final_norm.bin"))

    # 3. 按层（Layer）提取给 Node (纯 Raw Bytes)
    layer_indices = set()
    for k in all_tensors.keys():
        if "layers." in k:
            parts = k.split("layers.")
            if len(parts) > 1:
                layer_num = int(parts[1].split(".")[0])
                layer_indices.add(layer_num)

    sorted_layers = sorted(list(layer_indices))
    print(f"检测到总 Layer 层数: {len(sorted_layers)} (层索引: {sorted_layers})")

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
        
        # 将 Node 的层组合打包
        combined_bin = bytearray()
        for l in chunk_layers:
            print(f"  └─ ⚙️ 拼装第 {l} 层纯净 Raw Bytes (转 FP32)...")
            combined_bin.extend(export_raw_layer_bin(layer_bin_tensors, l))
            
        output_path = os.path.join(output_dir, bin_name)
        with open(output_path, 'wb') as f:
            f.write(combined_bin)
            
        print(f"✅ 生成 Node 固件: {bin_name} | 大小: {len(combined_bin) / (1024*1024):.2f} MB")

    print("\n🎉 所有文件拆分打包完成！")

if __name__ == "__main__":
    split_and_pack_safetensors()