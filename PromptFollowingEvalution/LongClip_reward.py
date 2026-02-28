"""
LongCLIP 图像-文本相似度 Reward 脚本 (参照 Qwen3_vl_reward)

功能:
  1. 从标注 TSV/CSV 读取 (Prompt, LocalPath, image4-promptFollowing)
  2. 用 LongCLIP 计算每对 (image, prompt) 的余弦相似度作为 score
  3. 计算 AUC (Good vs Bad) 以及序数相关性 (Bad < Fair < Good)

快速开始:
  pip install torch torchvision pillow pandas tqdm scikit-learn scipy
  git clone https://github.com/beichenzbc/Long-CLIP.git
  下载模型到 Long-CLIP/checkpoints/longclip-L.pt
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau

# ================= 配置 =================
LONGCLIP_REPO_PATH = "./Long-CLIP"
MODEL_PATH = "./Long-CLIP/checkpoints/longclip-L.pt"
INPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_LongCLIP_scores.tsv"
IMAGE_FOLDER = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
BATCH_SIZE = 64

# ================= 加载 LongCLIP =================
sys.path.insert(0, LONGCLIP_REPO_PATH)

try:
    from model import longclip
except ImportError:
    print(f"错误: 无法导入 longclip 模块")
    print(f"请确保已克隆 Long-CLIP 仓库到: {LONGCLIP_REPO_PATH}")
    print(f"运行: git clone https://github.com/beichenzbc/Long-CLIP.git")
    sys.exit(1)


# ================= 推理函数 =================
@torch.no_grad()
def compute_similarity_batch(model, preprocess, image_paths, prompts, device):
    """
    计算一批 (image, prompt) 对的余弦相似度
    返回: numpy array of shape (batch_size,)
    """
    images = []
    valid_indices = []
    for i, img_path in enumerate(image_paths):
        try:
            img = preprocess(Image.open(img_path)).unsqueeze(0)
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")

    if len(images) == 0:
        return np.array([]), []

    images_tensor = torch.cat(images, dim=0).to(device)

    # 逐条 tokenize 文本（每个 prompt 对应自己的图像）
    valid_prompts = [prompts[i] for i in valid_indices]
    text_tokens = longclip.tokenize(valid_prompts).to(device)

    image_features = model.encode_image(images_tensor)
    text_features = model.encode_text(text_tokens)

    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 逐对余弦相似度 (对角线)
    similarities = (image_features * text_features).sum(dim=-1).cpu().numpy()

    return similarities, valid_indices


# ================= 主程序 =================
def main():
    # 读取数据
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t', header=0)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    full_df['FullLocalPath'] = full_df['LocalPath'].apply(lambda x: os.path.join(IMAGE_FOLDER, x))
    valid_df = full_df[full_df['FullLocalPath'].apply(os.path.exists)].copy()
    print(f"总行数: {len(full_df)}, 有效样本数(图片存在): {len(valid_df)}")
    full_df = valid_df

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请从 https://huggingface.co/BeichenZhang/LongCLIP-L 下载 longclip-L.pt")
        sys.exit(1)

    # 设置设备 & 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载 LongCLIP 模型...")
    model, preprocess = longclip.load(MODEL_PATH, device=device)
    model.eval()

    # ================= 批量推理 =================
    final_results = []
    total = len(full_df)
    rows = list(full_df.iterrows())

    for start in tqdm(range(0, total, BATCH_SIZE), desc="推理中"):
        batch_rows = rows[start : start + BATCH_SIZE]
        image_paths = [r['FullLocalPath'] for _, r in batch_rows]
        prompts = [r['Prompt'] for _, r in batch_rows]

        try:
            scores, valid_indices = compute_similarity_batch(
                model, preprocess, image_paths, prompts, device
            )
            for j, vi in enumerate(valid_indices):
                res = batch_rows[vi][1].to_dict()
                res['clip_score'] = float(scores[j])
                final_results.append(res)
        except Exception as e:
            print(f"Batch 推理失败: {e}")
            continue

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"\n结果已保存到: {OUTPUT_TSV}")
    print(f"成功处理 {len(final_results)} / {total} 个样本")

    # ================= 评估指标 =================
    print("\n=== 相似度分布统计 ===")
    print(f"整体 clip_score: min={final_df['clip_score'].min():.3f}, "
          f"mean={final_df['clip_score'].mean():.3f}, "
          f"max={final_df['clip_score'].max():.3f}")

    good_scores = final_df[final_df['image4-promptFollowing'] == 'Good']['clip_score']
    fair_scores = final_df[final_df['image4-promptFollowing'] == 'fair']['clip_score']
    bad_scores  = final_df[final_df['image4-promptFollowing'] == 'Bad']['clip_score']

    print(f"\nBad样本  (n={len(bad_scores):4d}): mean={bad_scores.mean():.3f}, std={bad_scores.std():.3f}")
    print(f"Fair样本 (n={len(fair_scores):4d}): mean={fair_scores.mean():.3f}, std={fair_scores.std():.3f}")
    print(f"Good样本 (n={len(good_scores):4d}): mean={good_scores.mean():.3f}, std={good_scores.std():.3f}")
    print(f"\nGood-Bad差距:  {good_scores.mean() - bad_scores.mean():.3f}")
    print(f"Good-Fair差距: {good_scores.mean() - fair_scores.mean():.3f}")
    print(f"Fair-Bad差距:  {fair_scores.mean() - bad_scores.mean():.3f}")

    # ========== 评估1: 二分类 AUC (Good vs Bad) ==========
    print("\n" + "=" * 60)
    print("【评估1】二分类: Good vs Bad (排除Fair)")
    print("=" * 60)

    eval_df_binary = final_df[final_df['image4-promptFollowing'].isin(['Good', 'Bad'])].copy()
    y_true_binary = eval_df_binary['image4-promptFollowing'].map({'Good': 1, 'Bad': 0})
    y_score_binary = eval_df_binary['clip_score']

    auc_binary = roc_auc_score(y_true_binary, y_score_binary)
    good_count = sum(y_true_binary == 1)
    bad_count  = sum(y_true_binary == 0)
    print(f"样本数: {len(eval_df_binary)} (Good: {good_count}, Bad: {bad_count})")
    print(f"⭐ AUC: {auc_binary:.4f}")

    # 用中位数作为阈值做分类分析
    threshold = eval_df_binary['clip_score'].median()
    false_neg = eval_df_binary[(eval_df_binary['image4-promptFollowing'] == 'Good') &
                               (eval_df_binary['clip_score'] < threshold)]
    false_pos = eval_df_binary[(eval_df_binary['image4-promptFollowing'] == 'Bad') &
                               (eval_df_binary['clip_score'] > threshold)]

    print(f"\n以中位数 {threshold:.4f} 为阈值的分类性能:")
    print(f"  假阴性 (Good误判为Bad): {len(false_neg):3d} / {good_count} = {len(false_neg)/good_count*100:5.1f}%")
    print(f"  假阳性 (Bad误判为Good): {len(false_pos):3d} / {bad_count} = {len(false_pos)/bad_count*100:5.1f}%")
    print(f"  准确率: {(good_count + bad_count - len(false_neg) - len(false_pos)) / len(eval_df_binary) * 100:.1f}%")

    # ========== 评估2: 序数相关性 (Bad < Fair < Good) ==========
    print("\n" + "=" * 60)
    print("【评估2】三分类序数关系: Bad(0) < Fair(1) < Good(2)")
    print("=" * 60)

    eval_df_3class = final_df[final_df['image4-promptFollowing'].isin(['Good', 'Bad', 'fair'])].copy()
    y_true_ordinal  = eval_df_3class['image4-promptFollowing'].map({'Bad': 0, 'fair': 1, 'Good': 2})
    y_score_ordinal = eval_df_3class['clip_score']

    print(f"样本数: {len(eval_df_3class)} (Bad: {sum(y_true_ordinal==0)}, "
          f"Fair: {sum(y_true_ordinal==1)}, Good: {sum(y_true_ordinal==2)})")

    spearman_corr, spearman_p = spearmanr(y_true_ordinal, y_score_ordinal)
    kendall_corr, kendall_p   = kendalltau(y_true_ordinal, y_score_ordinal)

    print(f"⭐ Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"⭐ Kendall's Tau:    {kendall_corr:.4f} (p={kendall_p:.2e})")

    # Fair 样本的预测分布 (固定阈值 0.33/0.67，与 Qwen3_vl_reward 对齐)
    score_min = eval_df_3class['clip_score'].min()
    score_max = eval_df_3class['clip_score'].max()
    # 将 clip_score 归一化到 [0,1] 后与 0.33/0.67 比较
    norm_scores = (eval_df_3class['clip_score'] - score_min) / (score_max - score_min + 1e-8)
    fair_mask  = eval_df_3class['image4-promptFollowing'] == 'fair'
    fair_total = fair_mask.sum()

    fair_as_bad     = (fair_mask & (norm_scores < 0.33)).sum()
    fair_as_neutral = (fair_mask & (norm_scores >= 0.33) & (norm_scores <= 0.67)).sum()
    fair_as_good    = (fair_mask & (norm_scores > 0.67)).sum()

    print(f"\nFair样本的预测分布 (归一化后按0.33/0.67分割):")
    print(f"  倾向Bad  (norm<0.33): {fair_as_bad:3d} / {fair_total} = {fair_as_bad/fair_total*100:5.1f}%")
    print(f"  中立     (0.33-0.67): {fair_as_neutral:3d} / {fair_total} = {fair_as_neutral/fair_total*100:5.1f}%")
    print(f"  倾向Good (norm>0.67): {fair_as_good:3d} / {fair_total} = {fair_as_good/fair_total*100:5.1f}%")

    # ========== 保存错误样本 ==========
    print("\n" + "=" * 60)
    if len(false_neg) > 0:
        fn_file = OUTPUT_TSV.replace('.tsv', '_false_negative.csv')
        false_neg.nsmallest(20, 'clip_score')[['Prompt', 'LocalPath', 'clip_score']].to_csv(fn_file, index=False)
        print(f"✓ 已保存 {min(20, len(false_neg))} 个假阴性样本 → {fn_file}")

    if len(false_pos) > 0:
        fp_file = OUTPUT_TSV.replace('.tsv', '_false_positive.csv')
        false_pos.nlargest(20, 'clip_score')[['Prompt', 'LocalPath', 'clip_score']].to_csv(fp_file, index=False)
        print(f"✓ 已保存 {min(20, len(false_pos))} 个假阳性样本 → {fp_file}")

    print("=" * 60)
    print("\n完成!")


if __name__ == "__main__":
    main()