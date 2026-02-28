import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "google/siglip-so400m-patch14-384"  # 或本地路径
INPUT_TSV  = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_SigLIP_scores.tsv"
IMAGE_FOLDER = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
BATCH_SIZE = 32

# ================= 加载模型 =================
if accelerator.is_main_process:
    print(f"正在加载模型: {MODEL_PATH} ...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": accelerator.process_index},
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = accelerator.prepare(model)
model.eval()

if accelerator.is_main_process:
    print("模型加载完成")

# ================= 推理函数 =================
@torch.no_grad()
def compute_siglip_scores(image_paths, prompts):
    """
    计算一批 (image, prompt) 对的 SigLIP 相似度分数
    返回: numpy array of shape (valid_batch_size,), valid_indices list
    """
    images = []
    valid_indices = []
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")

    if len(images) == 0:
        return np.array([]), []

    valid_prompts = [prompts[i] for i in valid_indices]

    inputs = processor(
        images=images,
        text=valid_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(accelerator.device)

    # 转换为 bfloat16 与模型一致
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    outputs = model(**inputs)

    # logits_per_image: shape (n_images, n_texts)，取对角线得到配对得分
    logits = outputs.logits_per_image.diagonal().float().cpu().numpy()

    return logits, valid_indices


# ================= 主程序 =================
def main():
    # 读取数据
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t', header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    # 按进程分片
    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    my_df['FullLocalPath'] = my_df['LocalPath'].apply(lambda x: os.path.join(IMAGE_FOLDER, x))
    my_df = my_df[my_df['FullLocalPath'].apply(os.path.exists)].copy()

    if accelerator.is_main_process:
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    # ================= 批量推理 =================
    final_results = []
    rows = list(my_df.iterrows())
    total = len(rows)

    for start in tqdm(range(0, total, BATCH_SIZE),
                      desc=f"推理中 [rank {accelerator.process_index}]",
                      disable=not accelerator.is_main_process):
        batch_rows = rows[start : start + BATCH_SIZE]
        image_paths = [r['FullLocalPath'] for _, r in batch_rows]
        prompts     = [r['Prompt'] for _, r in batch_rows]

        try:
            scores, valid_indices = compute_siglip_scores(image_paths, prompts)
            for j, vi in enumerate(valid_indices):
                res = batch_rows[vi][1].to_dict()
                res['siglip_score'] = float(scores[j])
                final_results.append(res)
        except Exception as e:
            print(f"Batch 推理失败 (rank {accelerator.process_index}): {e}")
            continue

    # ================= 保存分片结果 =================
    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_output_file, sep='\t', index=False)

    accelerator.wait_for_everyone()

    # ================= 主进程合并 & 评估 =================
    if accelerator.is_main_process:
        print("\n>>> 正在合并结果并计算指标...")
        all_dfs = [pd.read_csv(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"), sep='\t')
                   for i in range(accelerator.num_processes)]
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f"成功处理 {len(final_df)} / {len(full_df)} 个样本")

        # ========== 分布统计 ==========
        print("\n=== 相似度分布统计 ===")
        print(f"整体 siglip_score: min={final_df['siglip_score'].min():.3f}, "
              f"mean={final_df['siglip_score'].mean():.3f}, "
              f"max={final_df['siglip_score'].max():.3f}")

        good_scores = final_df[final_df['image4-promptFollowing'] == 'Good']['siglip_score']
        fair_scores = final_df[final_df['image4-promptFollowing'] == 'fair']['siglip_score']
        bad_scores  = final_df[final_df['image4-promptFollowing'] == 'Bad']['siglip_score']

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
        y_true_binary  = eval_df_binary['image4-promptFollowing'].map({'Good': 1, 'Bad': 0})
        y_score_binary = eval_df_binary['siglip_score']

        auc_binary  = roc_auc_score(y_true_binary, y_score_binary)
        good_count  = sum(y_true_binary == 1)
        bad_count   = sum(y_true_binary == 0)
        print(f"样本数: {len(eval_df_binary)} (Good: {good_count}, Bad: {bad_count})")
        print(f"⭐ AUC: {auc_binary:.4f}")

        threshold = eval_df_binary['siglip_score'].median()
        false_neg = eval_df_binary[(eval_df_binary['image4-promptFollowing'] == 'Good') &
                                   (eval_df_binary['siglip_score'] < threshold)]
        false_pos = eval_df_binary[(eval_df_binary['image4-promptFollowing'] == 'Bad') &
                                   (eval_df_binary['siglip_score'] > threshold)]

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
        y_score_ordinal = eval_df_3class['siglip_score']

        print(f"样本数: {len(eval_df_3class)} (Bad: {sum(y_true_ordinal==0)}, "
              f"Fair: {sum(y_true_ordinal==1)}, Good: {sum(y_true_ordinal==2)})")

        spearman_corr, spearman_p = spearmanr(y_true_ordinal, y_score_ordinal)
        kendall_corr,  kendall_p  = kendalltau(y_true_ordinal, y_score_ordinal)

        print(f"⭐ Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"⭐ Kendall's Tau:    {kendall_corr:.4f} (p={kendall_p:.2e})")

        # Fair 样本的预测分布
        fair_as_bad     = sum((eval_df_3class['image4-promptFollowing'] == 'fair') &
                              (eval_df_3class['siglip_score'] < eval_df_3class['siglip_score'].quantile(0.33)))
        fair_as_neutral = sum((eval_df_3class['image4-promptFollowing'] == 'fair') &
                              (eval_df_3class['siglip_score'] >= eval_df_3class['siglip_score'].quantile(0.33)) &
                              (eval_df_3class['siglip_score'] <= eval_df_3class['siglip_score'].quantile(0.67)))
        fair_as_good    = sum((eval_df_3class['image4-promptFollowing'] == 'fair') &
                              (eval_df_3class['siglip_score'] > eval_df_3class['siglip_score'].quantile(0.67)))
        fair_total = sum(y_true_ordinal == 1)

        q33 = eval_df_3class['siglip_score'].quantile(0.33)
        q67 = eval_df_3class['siglip_score'].quantile(0.67)
        print(f"\nFair样本的预测分布 (按整体三分位 [{q33:.3f}, {q67:.3f}]):")
        print(f"  倾向Bad  (score<{q33:.3f}): {fair_as_bad:3d} / {fair_total} = {fair_as_bad/fair_total*100:5.1f}%")
        print(f"  中立     ({q33:.3f}-{q67:.3f}): {fair_as_neutral:3d} / {fair_total} = {fair_as_neutral/fair_total*100:5.1f}%")
        print(f"  倾向Good (score>{q67:.3f}): {fair_as_good:3d} / {fair_total} = {fair_as_good/fair_total*100:5.1f}%")

        # ========== 保存错误样本 ==========
        print("\n" + "=" * 60)
        if len(false_neg) > 0:
            fn_file = OUTPUT_TSV.replace('.tsv', '_false_negative.csv')
            false_neg.nsmallest(20, 'siglip_score')[['Prompt', 'LocalPath', 'siglip_score']].to_csv(fn_file, index=False)
            print(f"✓ 已保存 {min(20, len(false_neg))} 个假阴性样本 → {fn_file}")

        if len(false_pos) > 0:
            fp_file = OUTPUT_TSV.replace('.tsv', '_false_positive.csv')
            false_pos.nlargest(20, 'siglip_score')[['Prompt', 'LocalPath', 'siglip_score']].to_csv(fp_file, index=False)
            print(f"✓ 已保存 {min(20, len(false_pos))} 个假阳性样本 → {fp_file}")

        print("=" * 60)

        # 保存完整结果
        final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
        print(f"\n结果已保存到: {OUTPUT_TSV}")

        # 清理临时文件
        for i in range(accelerator.num_processes):
            tmp = OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")
            if os.path.exists(tmp):
                os.remove(tmp)

        print("\n完成!")


if __name__ == "__main__":
    main()
