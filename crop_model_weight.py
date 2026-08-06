import os
import json
import torch
from safetensors.torch import load_file, save_file

input_safetensor = "./Qwen2-0.5B/model.safetensors"
output_dir = "./cropped_model"
output_safetensor = os.path.join(output_dir, "model.safetensors")

TARGET_VOCAB_SIZE = 32000

# Qwen2 关键的特殊 Token 原始高位 ID 清单
# 151643: <|endoftext|>, 151644: <|im_start|>, 151645: <|im_end|> ...
SPECIAL_TOKENS_MAPPING = {
    151643: 31997,  # bos / eos (<|endoftext|>)
    151644: 31998,  # <|im_start|>
    151645: 31999,  # <|im_end|>
}

print("正在读取原始 safetensors 文件...")
state_dict = load_file(input_safetensor)

if "model.embed_tokens.weight" in state_dict:
    old_embed = state_dict["model.embed_tokens.weight"]
    
    # 1. 先截取前 32000 行
    new_embed = old_embed[:TARGET_VOCAB_SIZE, :].clone()
    
    # 2. 关键补丁：把高位特殊 Token 的原始 Embedding 向量，强行覆盖写入新末尾 ID 上
    for old_id, new_id in SPECIAL_TOKENS_MAPPING.items():
        if old_id < old_embed.shape[0]:
            print(f"搬运物理向量: 原始 ID {old_id} -> 新物理 ID {new_id}")
            new_embed[new_id] = old_embed[old_id]
            
    state_dict["model.embed_tokens.weight"] = new_embed

# 3. 删掉旧的 lm_head (共享物理内存)
if "lm_head.weight" in state_dict:
    del state_dict["lm_head.weight"]

os.makedirs(output_dir, exist_ok=True)
save_file(state_dict, output_safetensor)
print(f"✅ 带物理向量重映射的权重已保存至: {output_safetensor}")