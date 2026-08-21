from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../cropped_Qwen")
demo_text = "Q: Who are you?\nA: I am a 1.58-bit LLM running on a ESP32 cluster."
print("标准输入:", tokenizer(demo_text)["input_ids"])