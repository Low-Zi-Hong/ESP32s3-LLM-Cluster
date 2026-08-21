import os
import json

BASE_DIR = "../Qwen2-0.5B"
OUTPUT_DIR = "../cropped_Qwen"
TARGET_VOCAB_SIZE = 32000
REGULAR_VOCAB_SIZE = 31990

# 特殊 Token 显式物理 ID 映射
SPECIAL_MAP = {
    "<|endoftext|>": 31997,
    "<|im_start|>": 31998,
    "<|im_end|>": 31999
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载原始 tokenizer.json
tok_json_path = os.path.join(BASE_DIR, "tokenizer.json")
with open(tok_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

old_vocab = data["model"]["vocab"]

# 2. 截取前 31990 个普通 Token（排除特殊 Token 避免重复）
sorted_vocab = sorted(old_vocab.items(), key=lambda x: x[1])
new_vocab = {}

for token_str, _ in sorted_vocab:
    if token_str in SPECIAL_MAP:
        continue
    new_vocab[token_str] = len(new_vocab)
    if len(new_vocab) >= REGULAR_VOCAB_SIZE:
        break

# 3. 强行将特殊 Token 插入 vocab
for sp_token, new_id in SPECIAL_MAP.items():
    new_vocab[sp_token] = new_id

# 4. 过滤 Merges
old_merges = data["model"].get("merges", [])
new_merges = []
for merge_pair in old_merges:
    parts = merge_pair.split(" ")
    if len(parts) == 2:
        p1, p2 = parts[0], parts[1]
        if (p1 + p2) in new_vocab and p1 in new_vocab and p2 in new_vocab:
            new_merges.append(merge_pair)

data["model"]["vocab"] = new_vocab
data["model"]["merges"] = new_merges

# 5. 重构 added_tokens 数组 (这是 Fast Tokenizer 正确识别特殊符号的关键!)
new_added_tokens = []
added_tokens_decoder = {}

for token_str, new_id in SPECIAL_MAP.items():
    # 构造标准 added_token 对象
    item = {
        "id": new_id,
        "content": token_str,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True  # 必须为 True，强制匹配
    }
    new_added_tokens.append(item)
    added_tokens_decoder[str(new_id)] = item

data["added_tokens"] = new_added_tokens

# 6. 保存新的 tokenizer.json
out_tok_json = os.path.join(OUTPUT_DIR, "tokenizer.json")
with open(out_tok_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# 7. 更新 tokenizer_config.json
tok_cfg_path = os.path.join(BASE_DIR, "tokenizer_config.json")
if os.path.exists(tok_cfg_path):
    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    
    # 更新 added_tokens_decoder 保证加载成功
    tok_cfg["added_tokens_decoder"] = added_tokens_decoder
    tok_cfg["bos_token"] = "<|endoftext|>"
    tok_cfg["eos_token"] = "<|endoftext|>"
    
    out_tok_cfg = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
    with open(out_tok_cfg, "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=False, indent=2)

print("✅ Tokenizer 重新生成成功！")