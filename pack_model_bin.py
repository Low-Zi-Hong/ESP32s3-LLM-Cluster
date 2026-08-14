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
    
    def append_tensor(key_suffix, is_float=True, default_shape=None, default_val=0.0):
        # 灵活匹配后缀
        matched_key = None
        for k in layer_tensors.keys():
            if k.endswith(key_suffix):
                matched_key = k
                break
        
        if not matched_key:
            # 如果是 RMSNorm 找不到（被融合了），直接跳过，不写入任何数据！
            if "layernorm" in key_suffix:
                return 
            # 如果是 K 和 V 没有 Bias（某些架构里 K/V 不带 bias），用 0.0 占位
            elif "bias" in key_suffix:
                dim = 128 if ("k_proj" in key_suffix or "v_proj" in key_suffix) else 896
                final_bin.extend(torch.zeros(dim, dtype=torch.float32).numpy().tobytes())
                return
            else:
                raise KeyError(f"❌ 找不到关键权重后缀: {key_suffix} ...")

        tensor = layer_tensors[matched_key]
        
        if is_float:
            # 强转为 C++ 兼容的 32-bit 标准浮点数
            tensor = tensor.to(torch.float32)
        
        final_bin.extend(tensor.numpy().tobytes())

    # =================================================================
    # 严格按照 C++ 代码里的挂载顺序拼接 Bytes！
    # 适配了 weight_packed 和 gamma 的新命名，缺失的 Norm 用 1.0 补全
    # =================================================================
    
    # 1. Attention 前的 RMSNorm (如果缺失，用 896 维的 1.0 占位)
    append_tensor("input_layernorm.weight", is_float=True, default_shape=896, default_val=1.0)
    
    # 2. Q proj
    append_tensor("self_attn.q_proj.weight_packed", is_float=False) 
    append_tensor("self_attn.q_proj.gamma", is_float=True)
    append_tensor("self_attn.q_proj.bias", is_float=True, default_shape=896, default_val=0.0)
    
    # 3. K proj (GQA 维度 128)
    append_tensor("self_attn.k_proj.weight_packed", is_float=False)
    append_tensor("self_attn.k_proj.gamma", is_float=True)
    append_tensor("self_attn.k_proj.bias", is_float=True, default_shape=128, default_val=0.0)
    
    # 4. V proj (GQA 维度 128)
    append_tensor("self_attn.v_proj.weight_packed", is_float=False)
    append_tensor("self_attn.v_proj.gamma", is_float=True)
    append_tensor("self_attn.v_proj.bias", is_float=True, default_shape=128, default_val=0.0)
    
    # 5. O proj
    append_tensor("self_attn.o_proj.weight_packed", is_float=False)
    append_tensor("self_attn.o_proj.gamma", is_float=True)
    
    # 6. MLP 前的 RMSNorm (如果缺失，用 896 维的 1.0 占位)
    append_tensor("post_attention_layernorm.weight", is_float=True, default_shape=896, default_val=1.0)
    
    # 7. Gate proj
    append_tensor("mlp.gate_proj.weight_packed", is_float=False)
    append_tensor("mlp.gate_proj.gamma", is_float=True)
    
    # 8. Up proj
    append_tensor("mlp.up_proj.weight_packed", is_float=False)
    append_tensor("mlp.up_proj.gamma", is_float=True)
    
    # 9. Down proj
    append_tensor("mlp.down_proj.weight_packed", is_float=False)
    append_tensor("mlp.down_proj.gamma", is_float=True)

    return final_bin

def split_and_pack_safetensors(
    safetensors_path="./cropped_model/model158_bit4.safetensors",
    output_dir="./cropped_model/bins",
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