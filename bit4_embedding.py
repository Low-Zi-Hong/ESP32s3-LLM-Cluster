import torch
from safetensors import safe_open
from safetensors.torch import save_file

def pack_tensor_to_int4(weight: torch.Tensor):
    """
    将 FP16 张量按行量化为 4-bit，并打包成 uint8
    """
    max_val = weight.abs().max(dim=-1, keepdim=True).values
    scale = max_val / 7.0 + 1e-7
    weight_q = torch.clamp(torch.round(weight / scale), min=-8, max=7).to(torch.int8)
    weight_u4 = (weight_q + 8).to(torch.uint8)
    
    out_features, in_features = weight_u4.shape
    weight_reshaped = weight_u4.view(out_features, in_features // 2, 2)
    packed_weight = (weight_reshaped[:, :, 0] << 4) | weight_reshaped[:, :, 1]
    
    return packed_weight.cpu(), scale.to(torch.float16).cpu()

def repack_safetensors_with_4bit_emb_and_drop_head(
    input_path="cropped_model/model_158.safetensors", 
    output_path="cropped_model/model158_bit4.safetensors"
):
    print(f"正在打开文件: {input_path}")
    export_dict = {}
    
    # 1. 逐个读取原 safetensors 里的权重
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            
            # 2. 暴力剔除 LM Head 输出层 (释放 54.69 MB 冗余)
            if "lm_head.weight" in key or key.startswith("lm_head"):
                print(f"🗑️ 已剔除冗余输出层: {key}")
                continue
                
            tensor = f.get_tensor(key)
            
            # 3. 拦截 Embedding 层进行 4-bit 压缩 (压缩至约 13.67 MB)
            if "embed_tokens.weight" in key:
                print(f"\n🔍 找到 Embedding 层: {key} | 原始尺寸: {tensor.shape} ({tensor.dtype})")
                
                packed_emb, emb_scale = pack_tensor_to_int4(tensor)
                
                export_dict[key.replace(".weight", ".weight_packed_4bit")] = packed_emb
                export_dict[key.replace(".weight", ".scales")] = emb_scale
                
                print(f"✅ 4-bit 打包完成!")
                print(f"   -> {key.replace('.weight', '.weight_packed_4bit')} | 尺寸: {packed_emb.shape}")
                print(f"   -> {key.replace('.weight', '.scales')} | 尺寸: {emb_scale.shape}\n")
                
            else:
                # 4. 其他已经被处理过的 2-bit 层和 Norm 层原样保留
                export_dict[key] = tensor
                
    # 5. 写入最终针对 ESP32 优化的 safetensors
    print(f"📦 正在写入最终模型文件至: {output_path} ...")
    save_file(export_dict, output_path)
    print("🎉 转换完成！目前文件体积已达到流水线部署的理论极限。")

if __name__ == "__main__":
    repack_safetensors_with_4bit_emb_and_drop_head()