import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

block_size = 512              # 你的 ESP32-S3 上下文窗口长度
batch_size = 4                # 训练批次大小

class BitNet158STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        eps = 1e-8

        gamma = weight.abs().mean()

        weight_scaled = weight / (gamma + eps)

        weight_clip = torch.clamp(torch.round(weight_scaled),min = -1.0, max = 1.0)

        weight_quant = weight_clip * gamma

        ctx.save_for_backward(weight_scaled)

        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        weight_scaled, = ctx.saved_tensors

        grad_weight = grad_output.clone()

        grad_weight[weight_scaled > 1.0] = 0.0

        grad_weight[weight_scaled < -1.0 ] = 0.0

        return grad_weight

class BitNetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        
    def forward(self, x):
        # 将原始的浮点权重 (self.weight) 送入 STE 算子
        # 吐出 1.58-bit 的量化权重 (w_quant) 参与实际计算
        w_quant = BitNet158STE.apply(self.weight)
        return F.linear(x, w_quant, self.bias)

def replace_linear_with_bitnet(model, target_modules=["gate_proj", "up_proj", "down_proj"]):
    """
    递归遍历模型，将目标名称的 nn.Linear 替换为 BitNetLinear
    """
    for name, module in model.named_children():
        # 检查当前模块是否是我们要替换的目标，并且是 nn.Linear
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            
            # 1. 记录原层的形状和配置
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            # 2. 实例化你写的 QAT 层
            new_module = BitNetLinear(in_features, out_features, bias=has_bias)
            
            # 3. 灵魂转移：将 Qwen 的原始 FP32/FP16 权重拷贝给 QAT 层作为幕后真身
            new_module.weight.data = module.weight.data.clone()
            if has_bias:
                new_module.bias.data = module.bias.data.clone()
                
            # 4. 执行替换
            setattr(model, name, new_module)
            print(f"Replaced {name} with BitNetLinear")
            
        else:
            # 递归进入下一层
            replace_linear_with_bitnet(module, target_modules)
            
    return model

def export_158bit_safetensors(model, export_path="qwen_158bit_packed.safetensors"):
    model.eval() 
    export_dict = {}
    
    # 强制在 CPU 上进行打包计算，避免显存溢出或设备不匹配问题
    pack_multiplier = torch.tensor([64, 16, 4, 1], dtype=torch.uint8, device="cpu")
    
    with torch.no_grad():
        for name, module in model.named_modules():
            
            if isinstance(module, BitNetLinear):
                # 将权重移动到 CPU 进行处理
                weight = module.weight.data.cpu()
                
                gamma = weight.abs().mean()
                weight_scaled = weight / (gamma + 1e-8)
                weight_ternary = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)
                
                weight_mapped = (weight_ternary + 1.0).to(torch.uint8)
                
                out_features, in_features = weight_mapped.shape
                # ESP32-S3 的内存对齐通常要求维度是偶数，4的倍数是最好的
                assert in_features % 4 == 0, f"输入特征维度 {in_features} 必须能被 4 整除才能打包"
                
                weight_reshaped = weight_mapped.view(out_features, in_features // 4, 4)
                
                # 在 CPU 上完成张量乘法和求和打包
                weight_packed = (weight_reshaped * pack_multiplier).sum(dim=-1).to(torch.uint8)
                
                export_dict[f"{name}.weight_packed"] = weight_packed
                # 显式转换为 float16 供单片机读取
                export_dict[f"{name}.gamma"] = gamma.to(torch.float16)
                
                if module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)
                
                print(f"✅ 打包成功: {name} | 原始尺寸: {weight.shape} -> 压缩尺寸: {weight_packed.shape}")
                
            elif isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.Linear)):
                # 这里我们假设如果你使用了 Qwen2RMSNorm，你已经在外部 import 或定义了它
                # 但出于兼容性，我们通过属性检查来规避类名不匹配的问题
                if hasattr(module, 'weight') and module.weight is not None:
                    export_dict[f"{name}.weight"] = module.weight.data.cpu().to(torch.float16)
                if hasattr(module, 'bias') and module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)
                     
    from safetensors.torch import save_file # 如果没有在头部导入，这里也可以
    save_file(export_dict, export_path)
    print(f"\n🎉 导出完成！已保存极限压缩模型至: {export_path}")

if __name__ == '__main__':
    # 1. 直接指向你裁剪后的本地目录
    tokenizer = AutoTokenizer.from_pretrained("./cropped_model")

    # 2. 模拟 BPE 分词 (喂给 QAT 训练的数据流)
    text = "Hello, world! 这是一个测试。"
    tokens = tokenizer(text, return_tensors="pt")

    print(tokens["input_ids"]) 
    # 输出的 ID 应该都在 0~31999 之间，并且完美包含你的 <|im_start|> 等特殊 Token

    model = AutoModelForCausalLM.from_pretrained(
        "./cropped_model",
        dtype=torch.float16,
        device_map="auto" # 如果你有 GPU，它会自动放进 GPU
    )

    print(model)

    full_target_modules = [
        "gate_proj", "up_proj", "down_proj", # MLP层
        "q_proj", "k_proj", "v_proj", "o_proj" # Attention层
    ]

    model = replace_linear_with_bitnet(model, target_modules=full_target_modules)

    #get dataset


    # wikitext-2-raw-v1 是纯文本版本，体积小，适合跑通流程
    raw_datasets = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        # 直接对文本进行编码，不需要加特殊 prompt
        return tokenizer(examples["text"])

    # 使用 batched=True 加速处理，并清理掉原始的字符串文本列
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # 4. 第二阶段：拼接与定长分块 (Chunking)
    def group_texts(examples):
        # 将一个 batch 内所有的 input_ids (和 attention_mask) 首尾拼接成一维长数组
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        
        # 获取拼接后的总长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        block_size = 512              # 你的 ESP32-S3 上下文窗口长度
        batch_size = 4                # 训练批次大小
        
        # 丢弃最后不足一个 block_size 的余数部分，保证所有输入绝对等长
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # 按 block_size 进行精准切分
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # 核心：因果语言模型中，标签 (labels) 与输入 (input_ids) 完全一致
        # 模型的内部逻辑会自动将 logits 偏移一位去对齐计算 Loss
        result["labels"] = result["input_ids"].copy()
        
        return result

    # 执行分块，开启多进程加速
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        #num_proc=10, # 根据你的 CPU 核心数调整
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # 5. 转换为 PyTorch 张量并构建 DataLoader
    lm_datasets.set_format(type="torch")

    # 提取训练集并打乱顺序
    train_dataloader = DataLoader(
        lm_datasets["train"], 
        shuffle=True, 
        batch_size=batch_size
    )

    # 验证输出
    for batch in train_dataloader:
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Labels shape:", batch["labels"].shape)
        break



    # 基础超参数
    base_lr = 1e-5         # 普通层学习率
    bitnet_lr = 5e-6       # 1.58-bit 层学习率 (必须更小)
    weight_decay = 0.01    # 普通层的正则化惩罚

    bitnet_params = []
    standard_params = []

    # 遍历所有参数，根据它们是否属于 BitNetLinear 进行分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 我们之前定义的 1.58-bit 层只有 weight 和 bias
        # 根据你的类名判断，或者简单地用名称过滤
        if "BitNetLinear" in str(type(model.get_submodule(".".join(name.split(".")[:-1])))):
            bitnet_params.append(param)
            print(f"[QAT] 划入 BitNet 分组 (No Weight Decay): {name}")
        else:
            standard_params.append(param)

    # 将两组参数送入优化器
    optimizer = optim.AdamW([
        {"params": standard_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": bitnet_params, "lr": bitnet_lr, "weight_decay": 0.0} # 核心：关闭 1.58-bit 层的权重衰减
    ])

    # 设定训练轮数
    epochs = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    print("\n🚀 开始 1.58-bit 量化感知训练...")

    for epoch in range(epochs):
        total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # 1. 将数据移至 GPU
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 2. 清空上一步的梯度
            optimizer.zero_grad()
            
            # 3. 前向传播
            # CausalLM 架构下，只要传了 labels，模型内部会自动计算 CrossEntropyLoss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # 4. 反向传播 (此时会触发你手写的 BitNet158STE.backward)
            loss.backward()
            
            # 5. 梯度裁剪 (极度重要：防止突发的大梯度破坏 [-1.0, 1.0] 的边界)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. 更新幕后真身 (Latent Weights)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印日志
            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")
                
        print(f"✅ Epoch {epoch} 结束 | 平均 Loss: {total_loss / len(train_dataloader):.4f}")



    print("\n📦 训练结束，开始剥离量化权重并进行 2-bit 极限打包...")


    # 执行导出
    export_158bit_safetensors(model)