from transformers import AutoTokenizer

try:
    # 1. 尝试加载
    tokenizer = AutoTokenizer.from_pretrained("./cropped_model", trust_remote_code=True)
    print("✅ Tokenizer 加载成功！")
    print(f"实际 Vocabulary 大小: {len(tokenizer)}")

    # 2. 边缘测试文本（涵盖英文、基础中文、标点、数字、特殊符号）
    test_texts = [
        "Hello, world!",
        "ESP32-S3 嵌入式 AI 引擎测试。",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890 + - * / = %",
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    ]

    print("\n--- 开始编解码正确性测试 ---")
    for text in test_texts:
        # 编码
        tokens = tokenizer.encode(text)
        # 检查是否有超过 32000 的越界 ID
        max_id = max(tokens) if tokens else 0
        # 解码
        decoded = tokenizer.decode(tokens)
        
        print(f"原始输入: {text}")
        print(f"Token IDs (Max ID: {max_id}): {tokens}")
        print(f"解码还原: {decoded}")
        print("-" * 50)
        
        if max_id >= 32000:
            print(f"❌ 警告：检测到超出 32000 范围的 Token ID: {max_id}！")

except Exception as e:
    print("❌ 验证过程报错:", e)