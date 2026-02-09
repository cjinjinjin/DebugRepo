import os
import torch
from PIL import Image
from model import longclip  # 从克隆的仓库中导入 longclip 模块

# ================= 配置区域 =================
# 模型权重路径
MODEL_PATH = './longclip-L.pt' 

# 你的图片文件夹路径
IMAGE_FOLDER = './my_images' 

# 你的文本提示 (LongCLIP 支持超过 77 个 Token 的长文本)
TEXT_PROMPT = "A detailed photo of a futuristic city with flying cars and neon lights"

# 支持的图片格式
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
# ===========================================

def main():
    # 1. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型和预处理工具
    print("Loading LongCLIP model...")
    # 注意：这里使用的是 longclip.load 而不是 clip.load
    model, preprocess = longclip.load(MODEL_PATH, device=device)
    model.eval()

    # 3. 处理文本
    # LongCLIP 的核心优势：tokenize 支持长文本
    print(f"Processing prompt: '{TEXT_PROMPT}'")
    text_tokens = longclip.tokenize([TEXT_PROMPT]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # 归一化文本特征
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 4. 遍历文件夹并计算相关性
    results = []
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    print("Scanning images...")
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(VALID_EXTENSIONS):
            file_path = os.path.join(IMAGE_FOLDER, filename)
            
            try:
                # 加载并预处理图片
                image = Image.open(file_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)

                # 提取图片特征
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    # 归一化图片特征
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # 5. 计算相似度 (Cosine Similarity)
                # 计算公式: Similarity = (Image_Features · Text_Features)
                similarity = (100.0 * image_features @ text_features.T).item()
                
                results.append((filename, similarity))
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # 6. 排序并输出结果
    # 按相似度从高到低排序
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n====== Results (Ranked by Relevance) ======")
    print(f"{'Filename':<30} | {'Score':<10}")
    print("-" * 45)
    for filename, score in results:
        print(f"{filename:<30} | {score:.4f}")

if __name__ == "__main__":
    main()