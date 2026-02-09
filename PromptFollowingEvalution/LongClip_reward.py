"""
简化版 LongCLIP 图像文本相似度计算脚本

快速开始:
1. pip install torch torchvision pillow pandas tqdm
2. git clone https://github.com/beichenzbc/Long-CLIP.git
3. 下载模型到 Long-CLIP/checkpoints/longclip-L.pt
4. 修改下面的配置并运行
"""

import os
import sys
import torch
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==================== 配置区域 ====================

# LongCLIP 仓库路径
LONGCLIP_REPO_PATH = "./Long-CLIP"

# 图像文件夹路径
IMAGE_FOLDER = "./images"

# 文本提示词
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog", 
    "a beautiful landscape",
    "a person reading a book",
    "a modern building"
]

# 模型路径
MODEL_PATH = "./Long-CLIP/checkpoints/longclip-L.pt"

# 输出文件
OUTPUT_CSV = "./similarity_results.csv"

# ==================================================

# 将 LongCLIP 添加到系统路径
sys.path.insert(0, LONGCLIP_REPO_PATH)

try:
    from model import longclip
except ImportError:
    print(f"错误: 无法导入 longclip 模块")
    print(f"请确保已克隆 Long-CLIP 仓库到: {LONGCLIP_REPO_PATH}")
    print(f"运行: git clone https://github.com/beichenzbc/Long-CLIP.git")
    sys.exit(1)


def main():
    # 检查路径
    if not os.path.exists(IMAGE_FOLDER):
        print(f"错误: 图像文件夹不存在: {IMAGE_FOLDER}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请从以下地址下载:")
        print("https://huggingface.co/BeichenZhang/LongCLIP-L/blob/main/longclip-L.pt")
        sys.exit(1)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载 LongCLIP 模型...")
    model, preprocess = longclip.load(MODEL_PATH, device=device)
    model.eval()
    
    # 获取图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"警告: 在 {IMAGE_FOLDER} 中没有找到图像文件")
        sys.exit(1)
    
    # 处理每张图像
    results = []
    
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 加载图像
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            # 编码文本
            text = longclip.tokenize(PROMPTS).to(device)
            
            # 计算相似度
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 余弦相似度
                similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            
            # 保存结果
            result = {
                'image_name': img_path.name,
                'image_path': str(img_path)
            }
            
            for i, (prompt, score) in enumerate(zip(PROMPTS, similarities)):
                result[f'prompt_{i+1}'] = prompt
                result[f'score_{i+1}'] = float(score)
            
            # 最佳匹配
            best_idx = similarities.argmax()
            result['best_match'] = PROMPTS[best_idx]
            result['best_score'] = float(similarities[best_idx])
            
            results.append(result)
            
        except Exception as e:
            print(f"\n处理 {img_path.name} 时出错: {e}")
            continue
    
    # 保存到 CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: {OUTPUT_CSV}")
    print(f"成功处理 {len(results)} 张图像")
    
    # 显示统计
    print("\n=== 平均相似度分数 ===")
    for i, prompt in enumerate(PROMPTS):
        avg_score = df[f'score_{i+1}'].mean()
        print(f"{prompt}: {avg_score:.4f}")
    
    # 显示前5个结果
    print("\n=== 前 5 个结果 ===")
    print(df[['image_name', 'best_match', 'best_score']].head().to_string(index=False))
    
    print("\n完成!")


if __name__ == "__main__":
    main()