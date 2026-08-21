import json
import struct

def get_unicode_to_bytes():
    """HuggingFace 标准的 byte-to-unicode 反向映射"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 建立反向字典: Unicode 字符 -> 原始单字节 (0~255)
    return {chr(c): b for b, c in zip(bs, cs)}

def build_esp32_tokenizer(json_path="../cropped_Qwen/tokenizer.json", bin_path="../cropped_Qwen/tokenizer.bin"):
    print(f"正在读取 {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = data['model']['vocab']
    merges = data['model'].get('merges', [])
    vocab_size = max(vocab.values()) + 1
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    # 拿到反向解码表
    unicode_to_bytes = get_unicode_to_bytes()

    print(f"词表大小: {vocab_size}")
    print(f"合并规则数: {len(merges)}")

    token_array = []
    string_pool = bytearray()

    # 1. 构建词元数组和字符串池
    for i in range(vocab_size):
        if i in id_to_token:
            token_str = id_to_token[i]
            
            # 【核心修复】：把 JSON 里的转义字符，精准还原为原始的 16 进制 Byte！
            token_bytes = bytearray()
            for char in token_str:
                if char in unicode_to_bytes:
                    token_bytes.append(unicode_to_bytes[char])
                else:
                    token_bytes.extend(char.encode('utf-8', errors='ignore'))
            
            token_bytes = bytes(token_bytes)
        else:
            token_bytes = b""

        offset = len(string_pool)
        length = len(token_bytes)
        
        token_array.append(struct.pack('<IHH', offset, length, 0))
        string_pool.extend(token_bytes)
        string_pool.append(0)

    # 2. 构建纯整数的合并规则表 (保持你的神级排序逻辑不变)
    merge_list = []
    for rank, merge_str in enumerate(merges):
        parts = merge_str.split(' ')
        if len(parts) == 2:
            left, right = parts
            if left in vocab and right in vocab:
                left_id = vocab[left]
                right_id = vocab[right]
                result_token = left + right
                if result_token in vocab:
                    result_id = vocab[result_token]
                    merge_list.append((left_id, right_id, result_id, rank))

    merge_list.sort(key=lambda x: (x[0] << 16) | x[1])

    merge_data = bytearray()
    for m in merge_list:
        merge_data.extend(struct.pack('<HHHH', m[0], m[1], m[2], m[3]))

    # 3. 写入二进制文件
    magic = b'ESP3'
    header = struct.pack('<4sIII', magic, vocab_size, len(merge_list), len(string_pool))

    with open(bin_path, 'wb') as f:
        f.write(header)
        for t in token_array:
            f.write(t)
        f.write(merge_data)
        f.write(string_pool)
        
    print(f"✅ 打包成功！已生成修复版二进制文件: {bin_path}")

if __name__ == "__main__":
    build_esp32_tokenizer()