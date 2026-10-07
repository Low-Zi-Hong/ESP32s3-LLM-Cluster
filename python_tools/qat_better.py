import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

# 国内镜像加速（视网络情况保留或注释）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 1. 核心 QAT 算子 (BitNet b1.58 STE)
# ==========================================
class BitNet158STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        eps = 1e-8
        gamma = weight.abs().mean()
        weight_scaled = weight / (gamma + eps)
        # 裁剪到 [-1, 1] 并四舍五入为 {-1, 0, 1}
        weight_clip = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)
        weight_quant = weight_clip * gamma
        ctx.save_for_backward(weight_scaled)
        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        weight_scaled, = ctx.saved_tensors
        grad_weight = grad_output.clone()
        # 截断梯度的边界效应
        grad_weight[weight_scaled > 1.0] = 0.0
        grad_weight[weight_scaled < -1.0] = 0.0
        return grad_weight

class BitNetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        
    def forward(self, x):
        w_quant = BitNet158STE.apply(self.weight)
        return F.linear(x, w_quant, self.bias)

# ==========================================
# 2. 算子替换与打包导出
# ==========================================
def replace_linear_with_bitnet(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            new_module = BitNetLinear(in_features, out_features, bias=has_bias)
            new_module.weight.data = module.weight.data.clone()
            if has_bias:
                new_module.bias.data = module.bias.data.clone()
                
            setattr(model, name, new_module)
            count += 1
        else:
            count += replace_linear_with_bitnet(module, target_modules)
    return count

def export_158bit_safetensors(model, export_path="../cropped_Qwen/qwen_158.safetensors"):
    model.eval()
    export_dict = {}
    pack_multiplier = torch.tensor([1, 4, 16, 64], dtype=torch.uint8, device="cpu")

    print("\n📦 开始提取与打包模型权重...")
    with torch.no_grad():
        for name, module in model.named_modules():
            # 1. 针对三值量化层打包 (2-bit packed)
            if isinstance(module, BitNetLinear):
                weight = module.weight.data.cpu()
                gamma = weight.abs().mean()
                weight_scaled = weight / (gamma + 1e-8)
                weight_ternary = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)

                weight_mapped = torch.zeros_like(weight_ternary, dtype=torch.uint8)
                weight_mapped[weight_ternary == 1] = 1   # 映射 1 -> 0b01
                weight_mapped[weight_ternary == -1] = 2  # 映射 -1 -> 0b10

                out_features, in_features = weight_mapped.shape
                assert in_features % 4 == 0, f"输入特征维度 {in_features} 必须能被 4 整除"

                weight_reshaped = weight_mapped.view(out_features, in_features // 4, 4)
                weight_packed = (weight_reshaped * pack_multiplier).sum(dim=-1).to(torch.uint8)

                export_dict[f"{name}.weight_packed"] = weight_packed
                export_dict[f"{name}.gamma"] = gamma.to(torch.float16)

                if module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)

                print(f"  [量化打包] {name} | 原始: {weight.shape} -> 压缩: {weight_packed.shape}")

            # 2. 针对 RMSNorm 层 (保持 FP16)
            elif "norm" in module.__class__.__name__.lower() and hasattr(module, 'weight') and module.weight is not None:
                export_dict[f"{name}.weight"] = module.weight.data.cpu().to(torch.float16)
                if hasattr(module, 'bias') and module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)
                print(f"  [辅助权重] RMSNorm: {name}")

            # 3. 针对 Embedding / LM Head
            elif isinstance(module, nn.Embedding):
                export_dict[f"{name}.weight"] = module.weight.data.cpu().to(torch.float16)
                print(f"  [辅助权重] Embedding: {name}")

    # 确保导出目录存在并保存
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    save_file(export_dict, export_path)
    print(f"\n🎉 导出完成！已成功写入: {export_path}")
    print(f"📊 总张量数: {len(export_dict)}")

# ==========================================
# 3. 验证评估函数
# ==========================================
def evaluate(model, val_loader, device, max_eval_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
            count += 1
            
    model.train()
    return total_loss / max(count, 1)

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("🚀 启动 BitNet 1.58-bit QAT 稳定训练流程...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_id = "../cropped_Qwen"
    export_path = "../cropped_Qwen/qwen_158.safetensors"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    )
    
    # 梯度检查点，降低显存开销
    model.gradient_checkpointing_enable()
    
    print("⚙️ 正在将线性层替换为 BitNet 1.58-bit 模拟量化层...")
    num_replaced = replace_linear_with_bitnet(model)
    print(f"✅ 成功替换 {num_replaced} 个线性层。")
    
    model.to(device)
    model.train()
    
    print("📚 加载 WikiText-2 数据集...")
    dataset_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"])
        
    block_size = 512
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # 处理训练集
    train_data = dataset_raw["train"].filter(lambda x: len(x["text"].strip()) > 0)
    train_tokenized = train_data.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing Train")
    train_grouped = train_tokenized.map(group_texts, batched=True, desc=f"Grouping Train ({block_size})")
    train_grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 处理验证集
    val_data = dataset_raw["validation"].filter(lambda x: len(x["text"].strip()) > 0)
    val_tokenized = val_data.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing Val")
    val_grouped = val_tokenized.map(group_texts, batched=True, desc=f"Grouping Val ({block_size})")
    val_grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    batch_size = 1
    grad_accum_steps = 4
    
    train_loader = DataLoader(train_grouped, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_grouped, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # 训练超参数
    max_steps = 800           # 真实更新步数（800 steps 足够三值网络平稳收敛）
    eval_every = 100          # 每 100 次参数更新评估一次验证集 Loss
    base_lr = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=max_steps
    )
    
    print(f"\n🔥 开始微调: 目标步数 {max_steps}，累加步数 {grad_accum_steps}...")
    
    step_count = 0
    accumulated_loss = 0.0
    progress_bar = tqdm(total=max_steps, desc="QAT Optimization Steps")

    micro_step = 0

    for batch in train_loader:
        if step_count >= max_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        accumulated_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            # 梯度裁剪防震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_count += 1
            progress_bar.update(1)
            
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'train_loss': f"{accumulated_loss * grad_accum_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            accumulated_loss = 0.0

            # 验证集监控
            if step_count % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                tqdm.write(f"\n📍 Step {step_count}/{max_steps} | 验证集 Loss: {val_loss:.4f} | 困惑度 (PPL): {torch.exp(torch.tensor(val_loss)):.2f}")

    progress_bar.close()
    print("\n✅ QAT 训练结束，开始打包权重！")

    # 执行导出
    export_158bit_safetensors(model, export_path)

if __name__ == "__main__":
    main()