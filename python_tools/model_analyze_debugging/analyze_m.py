import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def unpack_158bit(packed_weight, gamma):
    """
    将 2-bit 压缩的 uint8 矩阵解包回 FP16 的 -1, 0, 1，并乘上 gamma
    编码: 0→0, 1→+1, 2→-1
    顺序: 低2位是第0个数，高2位是第3个数 (对应 pack_multiplier = [1,4,16,64])
    """
    out_features, packed_in_features = packed_weight.shape

    def decode(bits):
        # bits: uint8 tensor, values in {0,1,2,3}
        val = torch.zeros_like(bits, dtype=torch.int8)
        val[bits == 1] = 1
        val[bits == 2] = -1
        # bits == 0 保持 0，bits == 3 理论不该出现，也保持 0
        return val

    w0 = decode((packed_weight)       & 0b11)   # 低位 = 第0个数
    w1 = decode((packed_weight >> 2)  & 0b11)
    w2 = decode((packed_weight >> 4)  & 0b11)
    w3 = decode((packed_weight >> 6)  & 0b11)   # 高位 = 第3个数

    unpacked_ternary = torch.stack([w0, w1, w2, w3], dim=-1).view(out_features, -1).to(torch.float16)

    return unpacked_ternary * gamma.to(torch.float16)

def unpack_4bit_emb(packed_weight, scales):
    """
    将 4-bit 压缩的 uint8 词表解包回 FP16
    """
    vocab_size, packed_hidden = packed_weight.shape
    
    # 提取高 4 位和低 4 位，减去 8 (恢复为 -8 到 7)
    w0 = ((packed_weight >> 4) & 0b1111).to(torch.int8) - 8
    w1 = ((packed_weight) & 0b1111).to(torch.int8) - 8
    
    # 恢复维度 [vocab_size, hidden_size]
    unpacked_q = torch.stack([w0, w1], dim=-1).view(vocab_size, -1).to(torch.float16)
    
    # 乘上每行的缩放系数 (广播机制)
    return unpacked_q * scales.to(torch.float16)

def load_and_generate(
    original_model_path="../../cropped_Qwen",
    safetensors_path="../../cropped_Qwen/qwen_158_int4.safetensors"
):
    print("⏳ 正在加载原始架构...")
    # 先加载原始的空壳架构
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path, 
        torch_dtype=torch.float16, 
        device_map="cpu" # 验证解包逻辑，用 CPU 即可
    )
    
    print(f"🔓 正在从 {safetensors_path} 解包注入全损权重...")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # 获取 safetensors 里所有的 key
        keys = f.keys()
        
        # 1. 恢复 4-bit Embedding
        if "model.embed_tokens.weight_packed_4bit" in keys:
            packed_emb = f.get_tensor("model.embed_tokens.weight_packed_4bit")
            scales = f.get_tensor("model.embed_tokens.scales")
            unpacked_emb = unpack_4bit_emb(packed_emb, scales)
            
            # 暴力覆盖原生模型的 Embedding
            model.model.embed_tokens.weight.data.copy_(unpacked_emb)
            
            # ✨ 核心操作：由于我们删除了 LM Head，这里将输出层直接绑定到解包后的 Embedding
            model.lm_head.weight = model.model.embed_tokens.weight
            print("✅ 4-bit 词表已解包，并成功挂载到 LM Head！")
            
        # 2. 遍历恢复所有 1.58-bit (2-bit packed) 的 Attention 和 MLP 层
        for name, module in model.named_modules():
            packed_key = f"{name}.weight_packed"
            gamma_key = f"{name}.gamma"
            bias_key = f"{name}.bias"
            weight_key = f"{name}.weight" # 原生 FP16 层
            
            # 如果是量化层
            if packed_key in keys:
                packed_w = f.get_tensor(packed_key)
                gamma = f.get_tensor(gamma_key)
                
                # 执行 2-bit 逆向解包
                unpacked_w = unpack_158bit(packed_w, gamma)
                module.weight.data.copy_(unpacked_w)
                
                if getattr(module, 'bias', None) is not None and bias_key in keys:
                    module.bias.data.copy_(f.get_tensor(bias_key))
            
            # 如果是原生保留的 FP16 层 (比如 LayerNorm)
            elif weight_key in keys and "embed_tokens" not in name and "lm_head" not in name:
                module.weight.data.copy_(f.get_tensor(weight_key))
                if getattr(module, 'bias', None) is not None and bias_key in keys:
                    module.bias.data.copy_(f.get_tensor(bias_key))

    print("🚀 解包完成！模型准备就绪，开启全损测试...\n")
    print("="*50)
    

# 过滤出所有属于 model.layers.0 的参数
    layer0_params = {
        name: param for name, param in model.named_parameters() 
        if "model.layers.2." in name
    }

    if not layer0_params:
        print("❌ 未找到 'model.layers.0'，请确认层级命名或模型配置。")
        return

    for name, param in layer0_params.items():
        # 去掉前缀，让输出更整洁
        short_name = name.replace("model.layers.0.", "")
        
        # 展平并提取前 10 个数值，转为 float 显示
        flattened = param.data.view(-1)
        preview_vals = [round(float(v), 5) for v in flattened[:10]]
        
        # 判断当前层是否为三值量化层，顺便展示三值化后的离散值
        is_bitnet = any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        print(f"\n📌 子模块: {short_name}")
        print(f"   Shape: {list(param.shape)} | Dtype: {param.dtype}")
        print(f"   原始连续权重 (前10): {preview_vals}")
        
        if is_bitnet and "weight" in name:
            gamma = param.data.abs().mean()
            ternary = torch.clamp(torch.round(param.data / (gamma + 1e-8)), -1.0, 1.0).view(-1)[:10]
            print(f"   量化 Scale (Gamma): {float(gamma):.6f}")
            print(f"   三值映射值 {{-1, 0, 1}} (前10): {[int(v) for v in ternary]}")

    print("\n" + "=" * 60)
    print("✅ Layer 0 参数提取完成。")



if __name__ == "__main__":
    load_and_generate()