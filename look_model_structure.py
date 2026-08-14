from safetensors import safe_open
import os

def inspect_safetensors_structure(file_path):
    print(f"🔍 正在解析文件: {os.path.basename(file_path)}\n")
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        
        # 过滤出 Layer 0 的数据
        layer_0_keys = [k for k in keys if "layers.0." in k]
        
        if not layer_0_keys:
            print("❌ 没有找到 layers.0. 相关的数据，请检查文件。")
            return

        print(f"==================================================")
        print(f" Layer 0 包含的 Tensor 数量: {len(layer_0_keys)}")
        print(f"==================================================")

        for k in layer_0_keys:
            tensor = f.get_tensor(k)
            shape = list(tensor.shape)
            numel = tensor.numel()
            dtype = str(tensor.dtype).split('.')[-1]
            
            print(f"🔑 Key: {k}")
            print(f"   ├─ Shape : {shape}")
            print(f"   ├─ Dtype : {dtype}")
            print(f"   └─ 元素数: {numel} 个")
            
            # 如果是只有 1 个元素的标量，或者极小的一维数组，直接把值打出来看看
            if numel <= 4:
                val_list = tensor.flatten().tolist()
                print(f"   └─ 数据值: {val_list}")
            
            print("-" * 50)

if __name__ == "__main__":
    safetensors_path = "./cropped_model/model158_bit4.safetensors"
    inspect_safetensors_structure(safetensors_path)