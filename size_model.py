import json
import struct
import os

def format_bytes(size):
    """将字节数格式化为人类可读的 MB/GB 单位"""
    for unit in ['Bytes', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def analyze_safetensors_memory(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 1. 提取并解析 Safetensors 的 JSON 头部
    with open(file_path, "rb") as f:
        header_length_bytes = f.read(8)
        header_length = struct.unpack('<Q', header_length_bytes)[0]
        header_string = f.read(header_length).decode('utf-8')
        header_data = json.loads(header_string)

    if "__metadata__" in header_data:
        del header_data["__metadata__"]

    # 2. 初始化统计字典
    stats = {
        "total_size": 0,
        "layers": {},       # 用于存储每一层的数据
        "embedding": 0,
        "lm_head": 0,
        "final_norm": 0,
        "other": 0
    }

    # 3. 遍历解析所有张量
    for key, value in header_data.items():
        # 通过 offsets 精确计算字节数
        offsets = value["data_offsets"]
        size_bytes = offsets[1] - offsets[0]
        stats["total_size"] += size_bytes

        # 归类逻辑
        if "model.layers." in key:
            # 提取具体的层索引号
            layer_idx = int(key.split("model.layers.")[1].split(".")[0])
            
            # 初始化该层的统计结构
            if layer_idx not in stats["layers"]:
                stats["layers"][layer_idx] = {
                    "total": 0, "attn_total": 0, "mlp_total": 0, "norm_total": 0,
                    "q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 0,
                    "gate_proj": 0, "up_proj": 0, "down_proj": 0
                }
            
            layer = stats["layers"][layer_idx]
            layer["total"] += size_bytes

            # 拆解 Attention 模块
            if "self_attn" in key:
                layer["attn_total"] += size_bytes
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if proj in key:
                        layer[proj] += size_bytes
            
            # 拆解 MLP 模块
            elif "mlp" in key:
                layer["mlp_total"] += size_bytes
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    if proj in key:
                        layer[proj] += size_bytes
            
            # 拆解层归一化
            elif "layernorm" in key or "norm" in key:
                layer["norm_total"] += size_bytes

        # 全局层归类
        elif "embed_tokens" in key:
            stats["embedding"] += size_bytes
        elif "lm_head" in key:
            stats["lm_head"] += size_bytes
        elif key == "model.norm.weight":
            stats["final_norm"] += size_bytes
        else:
            stats["other"] += size_bytes

    # 4. 打印报告
    print("=" * 50)
    print(f"📊 模型全局内存占用分析")
    print("=" * 50)
    print(f"总物理大小    : {format_bytes(stats['total_size'])}")
    print(f"Embedding 层  : {format_bytes(stats['embedding'])}")
    print(f"LM Head 输出层: {format_bytes(stats['lm_head'])}")
    print(f"最终全局 Norm : {format_bytes(stats['final_norm'])}")
    print(f"其他孤立参数  : {format_bytes(stats['other'])}")
    
    if stats["layers"]:
        print("\n" + "=" * 50)
        print(f"🔍 隐藏层结构分析 (以 Layer 0 为例)")
        print("=" * 50)
        
        l0 = stats["layers"][0]
        print(f"👉 单层总物理大小 (Total Layer Size): {format_bytes(l0['total'])}")
        
        print(f"\n  ├─ Attention 模块总计: {format_bytes(l0['attn_total'])}")
        print(f"  │   ├─ q_proj: {format_bytes(l0['q_proj'])}")
        print(f"  │   ├─ k_proj: {format_bytes(l0['k_proj'])}")
        print(f"  │   ├─ v_proj: {format_bytes(l0['v_proj'])}")
        print(f"  │   └─ o_proj: {format_bytes(l0['o_proj'])}")
        
        print(f"\n  ├─ MLP 模块总计: {format_bytes(l0['mlp_total'])}")
        print(f"  │   ├─ gate_proj: {format_bytes(l0['gate_proj'])}")
        print(f"  │   ├─ up_proj:   {format_bytes(l0['up_proj'])}")
        print(f"  │   └─ down_proj: {format_bytes(l0['down_proj'])}")
        
        print(f"\n  └─ 层归一化 (RMSNorm): {format_bytes(l0['norm_total'])}")
        
        print(f"\nℹ️  提示: 该模型总共有 {len(stats['layers'])} 个相同的隐藏层。")
        print(f"    所有隐藏层总计占比: {format_bytes(l0['total'] * len(stats['layers']))}")

# 替换为你实际的 safetensors 文件路径
model_path = "./cropped_model/model158_bit4.safetensors"
analyze_safetensors_memory(model_path)