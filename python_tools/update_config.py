import json
import os

def update_model_config(model_dir, new_vocab_size=32000):
    config_path = os.path.join("../Qwen2-0.5B", "config.json")
    write_path = os.path.join(model_dir,"config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ 找不到文件: {config_path}")
        return

    # 1. 加载现有的 config.json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("="*40)
    print(f"🔧 原始 Vocab Size : {config.get('vocab_size')}")
    print(f"🔧 原始 EOS Token ID : {config.get('eos_token_id')}")
    
    # 2. 核心覆盖：更新词表大小
    config["vocab_size"] = new_vocab_size

    # 3. 核心覆盖：重新映射 Special Tokens
    # 【注意】这里的值必须和你 crop_token.py 里映射的新 ID 保持绝对一致！
    # 假设你把 <|im_end|> 映射到了 31999 (即词表的最后一个位置)
    NEW_EOS_TOKEN_ID = 31999
    
    config["eos_token_id"] = NEW_EOS_TOKEN_ID
    
    # 如果你的模型有 bos 或者 pad token，也在这里一并修改
    if "bos_token_id" in config:
        config["bos_token_id"] = NEW_EOS_TOKEN_ID  # 根据实际情况修改
    if "pad_token_id" in config:
        config["pad_token_id"] = NEW_EOS_TOKEN_ID  # 根据实际情况修改

    os.makedirs(model_dir, exist_ok=True)
    # 4. 保存回写
    with open(write_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("="*40)
    print(f"✅ config.json 更新成功！")
    print(f"🎯 新 Vocab Size   : {config['vocab_size']}")
    print(f"🎯 新 EOS Token ID  : {config['eos_token_id']}")
    print(f"📂 保存路径         : {write_path}")
    print("="*40)

if __name__ == "__main__":
    # 指向你裁剪后模型所在的文件夹
    target_model_directory = "../cropped_Qwen"
    
    update_model_config(
        model_dir=target_model_directory, 
        new_vocab_size=32000
    )