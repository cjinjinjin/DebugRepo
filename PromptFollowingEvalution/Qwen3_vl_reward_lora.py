import torch
if not hasattr(torch, "compiler"):
    class MockCompiler:
        def is_compiling(self): return False
    torch.compiler = MockCompiler()
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from swift import Swift
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"
LORA_PATH  = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Qwen3-VL-8B-PromptFollowing-LoRA/v3-20260303-032527/checkpoint-100"
INPUT_TSV  = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/sft_data_v2/test_labels.csv"
OUTPUT_TSV = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/sft_data_v2/test_lora_logits.tsv"
image_folder = ""  # test_labels.csv 中 full_image_path 已是绝对路径
BATCH_SIZE = 16

# ================= 加载模型 =================
if accelerator.is_main_process:
    print(f"正在加载 base 模型: {MODEL_PATH} ...")
    print(f"正在加载 LoRA adapter: {LORA_PATH} ...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": accelerator.process_index},
    trust_remote_code=True
)
model = Swift.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

yes_tokens     = ["Yes", " Yes", "yes", " yes", "YES", " YES", "Ġyes", "ĠYes"]
no_tokens      = ["No", " No", "no", " no", "NO", " NO", "Ġno", "ĠNo"]
partial_tokens = ["Part", " Part", "Partial", " Partial", "Almost", " Almost"]

ids_yes     = [processor.tokenizer.convert_tokens_to_ids(t) for t in yes_tokens]
ids_no      = [processor.tokenizer.convert_tokens_to_ids(t) for t in no_tokens]
ids_partial = [processor.tokenizer.convert_tokens_to_ids(t) for t in partial_tokens]

YES_IDS     = [i for i in ids_yes     if i is not None and i >= 0]
NO_IDS      = [i for i in ids_no      if i is not None and i >= 0]
PARTIAL_IDS = [i for i in ids_partial if i is not None and i >= 0]

if accelerator.is_main_process:
    print(f"监控的 Yes Token IDs: {YES_IDS}")
    print(f"监控的 No Token IDs: {NO_IDS}")
    print(f"监控的 Partial Token IDs: {PARTIAL_IDS}")

# 验证 LoRA merge 是否真的生效（对比 base 模型第一层 q_proj 的期望值）
_w = dict(model.named_parameters())
_key = [k for k in _w if 'language_model' in k and 'q_proj' in k][0]
_checksum = _w[_key].data.float().mean().item()
if accelerator.is_main_process:
    print(f"[LoRA验证] {_key} mean={_checksum:.8f}  (base应为约-0.00001, merged应不同)")

model = accelerator.prepare(model)

# ================= 逻辑函数 =================

@torch.no_grad()
def run_inference_logits(msgs_batch):
    inputs = processor.apply_chat_template(
        msgs_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(accelerator.device)

    outputs = model(**inputs)
    last_token_logits = outputs.logits[:, -1, :]

    if not hasattr(run_inference_logits, "debug_count"):
        run_inference_logits.debug_count = 0
        run_inference_logits.top_tokens_collection = []

    if run_inference_logits.debug_count < 100 and accelerator.is_main_process:
        topk_vals, topk_indices = torch.topk(last_token_logits, 5, dim=-1)
        for idx in range(topk_indices.shape[0]):
            if run_inference_logits.debug_count >= 100:
                break
            tokens  = topk_indices[idx].cpu().tolist()
            decoded = processor.tokenizer.convert_ids_to_tokens(tokens)
            run_inference_logits.top_tokens_collection.append(decoded[0])
            if run_inference_logits.debug_count < 5:
                print(f"\n[Sample {run_inference_logits.debug_count}] Top-5 tokens: {decoded}")
                print(f"  Logits: {topk_vals[idx].float().cpu().numpy()}")
            run_inference_logits.debug_count += 1

    if run_inference_logits.debug_count == 100 and accelerator.is_main_process:
        from collections import Counter
        token_freq = Counter(run_inference_logits.top_tokens_collection)
        print("\n" + "="*50)
        print("前100个样本的Top-1 Token频率统计:")
        for token, count in token_freq.most_common(10):
            print(f"  '{token}': {count}次 ({count:.1f}%)")
        print("="*50 + "\n")
        run_inference_logits.debug_count += 1

    yes_logits     = last_token_logits[:, YES_IDS].max(dim=-1).values
    no_logits      = last_token_logits[:, NO_IDS].max(dim=-1).values
    partial_logits = last_token_logits[:, PARTIAL_IDS].max(dim=-1).values if len(PARTIAL_IDS) > 0 else torch.zeros_like(yes_logits)

    combined_logits = torch.stack([no_logits, partial_logits, yes_logits], dim=-1)
    weights = torch.tensor([0.0, 0.5, 1.0]).to(combined_logits.device)
    probs_weighted = torch.softmax(combined_logits, dim=-1)
    probs = (probs_weighted * weights).sum(dim=-1)

    return probs.float().cpu().numpy(), yes_logits.float().cpu().numpy(), no_logits.float().cpu().numpy()


@torch.no_grad()
def debug_generate(msgs_batch, num_samples=5):
    inputs = processor.apply_chat_template(
        msgs_batch[:num_samples],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(accelerator.device)

    generated_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )
    input_len = inputs['input_ids'].shape[1]
    generated_text = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
    for i, text in enumerate(generated_text):
        print(f"[Generate Sample {i}]: '{text}'")


# ================= 主程序 =================
def main():
    try:
        full_df = pd.read_csv(INPUT_TSV, sep=',', header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    my_df['FullLocalPath'] = my_df['full_image_path']
    my_df = my_df[my_df['FullLocalPath'].apply(os.path.exists)].copy()

    if accelerator.is_main_process:
        print(f"模式: LoRA finetune 模型 Logits 评测")
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    final_results = []
    buffer = []

    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        user_prompt = row['Prompt']
        image_path  = row['FullLocalPath']

        prompt = f"""Does this image match the description: "{user_prompt}"?
            Consider: Are the key objects present? Is the scene/composition correct? Are visual details accurate?
            Answer only "Yes" or "No"."""
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        buffer.append({"msg": msg, "row_data": row.to_dict()})

        if len(buffer) == BATCH_SIZE:
            msgs_to_run = [b["msg"] for b in buffer]
            try:
                if not hasattr(main, "debug_generated") and accelerator.is_main_process:
                    print("\n>>> 调试：查看实际生成文本...")
                    debug_generate(msgs_to_run, num_samples=3)
                    main.debug_generated = True
                yes_probs, y_logits, n_logits = run_inference_logits(msgs_to_run)
                for idx, prob in enumerate(yes_probs):
                    res = buffer[idx]["row_data"]
                    res["yes_prob"]    = float(prob)
                    res["yes_logit"]   = float(y_logits[idx])
                    res["no_logit"]    = float(n_logits[idx])
                    res["pred_answer"] = "yes" if prob > 0.5 else "no"
                    final_results.append(res)
            except Exception as e:
                print(f"Batch 推理失败: {e}")
            buffer = []

    if buffer:
        msgs_to_run = [b["msg"] for b in buffer]
        yes_probs, y_logits, n_logits = run_inference_logits(msgs_to_run)
        for idx, prob in enumerate(yes_probs):
            res = buffer[idx]["row_data"]
            res["yes_prob"]    = float(prob)
            res["yes_logit"]   = float(y_logits[idx])
            res["no_logit"]    = float(n_logits[idx])
            res["pred_answer"] = "yes" if prob > 0.5 else "no"
            final_results.append(res)

    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_output_file, sep='\t', index=False)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n>>> 正在合并结果并计算 AUC 指标...")
        all_dfs = [pd.read_csv(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"), sep='\t')
                   for i in range(accelerator.num_processes)]
        final_df = pd.concat(all_dfs, ignore_index=True)

        print("\n=== 概率分布统计 ===")
        print(f"整体yes_prob: min={final_df['yes_prob'].min():.3f}, "
              f"mean={final_df['yes_prob'].mean():.3f}, "
              f"max={final_df['yes_prob'].max():.3f}")

        good_probs = final_df[final_df['image4-promptFollowing']=='Good']['yes_prob']
        fair_probs = final_df[final_df['image4-promptFollowing']=='fair']['yes_prob']
        bad_probs  = final_df[final_df['image4-promptFollowing']=='Bad']['yes_prob']

        print(f"\nBad样本  (n={len(bad_probs):4d}): mean={bad_probs.mean():.3f}, std={bad_probs.std():.3f}")
        print(f"Fair样本 (n={len(fair_probs):4d}): mean={fair_probs.mean():.3f}, std={fair_probs.std():.3f}")
        print(f"Good样本 (n={len(good_probs):4d}): mean={good_probs.mean():.3f}, std={good_probs.std():.3f}")
        print(f"\nGood-Bad差距: {good_probs.mean() - bad_probs.mean():.3f}")
        print(f"Good-Fair差距: {good_probs.mean() - fair_probs.mean():.3f}")
        print(f"Fair-Bad差距: {fair_probs.mean() - bad_probs.mean():.3f}")

        print("\n" + "=" * 60)
        print("【评估1】二分类: Good vs Bad (排除Fair)")
        print("=" * 60)

        eval_df_binary = final_df[final_df['image4-promptFollowing'].isin(['Good', 'Bad'])].copy()
        y_true_binary  = eval_df_binary['image4-promptFollowing'].map({'Good': 1, 'Bad': 0})
        y_score_binary = eval_df_binary['yes_prob']

        auc_binary = roc_auc_score(y_true_binary, y_score_binary)
        good_count = sum(y_true_binary==1)
        bad_count  = sum(y_true_binary==0)
        print(f"样本数: {len(eval_df_binary)} (Good: {good_count}, Bad: {bad_count})")
        print(f"⭐ AUC: {auc_binary:.4f}")

        false_neg = eval_df_binary[(eval_df_binary['image4-promptFollowing']=='Good') & (eval_df_binary['yes_prob'] < 0.5)]
        false_pos = eval_df_binary[(eval_df_binary['image4-promptFollowing']=='Bad')  & (eval_df_binary['yes_prob'] > 0.5)]
        print(f"\n以0.5为阈值的分类性能:")
        print(f"  假阴性 (Good误判为Bad): {len(false_neg):3d} / {good_count} = {len(false_neg)/good_count*100:5.1f}%")
        print(f"  假阳性 (Bad误判为Good): {len(false_pos):3d} / {bad_count} = {len(false_pos)/bad_count*100:5.1f}%")
        print(f"  准确率: {(good_count + bad_count - len(false_neg) - len(false_pos)) / len(eval_df_binary) * 100:.1f}%")

        print("\n" + "=" * 60)
        print("【评估2】三分类序数关系: Bad(0) < Fair(1) < Good(2)")
        print("=" * 60)

        eval_df_3class = final_df[final_df['image4-promptFollowing'].isin(['Good', 'Bad', 'fair'])].copy()
        y_true_ordinal  = eval_df_3class['image4-promptFollowing'].map({'Bad': 0, 'fair': 1, 'Good': 2})
        y_score_ordinal = eval_df_3class['yes_prob']
        print(f"样本数: {len(eval_df_3class)} (Bad: {sum(y_true_ordinal==0)}, Fair: {sum(y_true_ordinal==1)}, Good: {sum(y_true_ordinal==2)})")

        from scipy.stats import spearmanr, kendalltau
        spearman_corr, spearman_p = spearmanr(y_true_ordinal, y_score_ordinal)
        kendall_corr,  kendall_p  = kendalltau(y_true_ordinal, y_score_ordinal)
        print(f"⭐ Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"⭐ Kendall's Tau:    {kendall_corr:.4f} (p={kendall_p:.2e})")

        fair_total      = sum(y_true_ordinal==1)
        fair_as_bad     = sum((eval_df_3class['image4-promptFollowing']=='fair') & (eval_df_3class['yes_prob'] < 0.33))
        fair_as_neutral = sum((eval_df_3class['image4-promptFollowing']=='fair') & (eval_df_3class['yes_prob'] >= 0.33) & (eval_df_3class['yes_prob'] <= 0.67))
        fair_as_good    = sum((eval_df_3class['image4-promptFollowing']=='fair') & (eval_df_3class['yes_prob'] > 0.67))
        print(f"\nFair样本的预测分布:")
        print(f"  倾向Bad  (prob<0.33): {fair_as_bad:3d} / {fair_total} = {fair_as_bad/fair_total*100:5.1f}%")
        print(f"  中立     (0.33-0.67): {fair_as_neutral:3d} / {fair_total} = {fair_as_neutral/fair_total*100:5.1f}%")
        print(f"  倾向Good (prob>0.67): {fair_as_good:3d} / {fair_total} = {fair_as_good/fair_total*100:5.1f}%")

        print("\n" + "=" * 60)
        if len(false_neg) > 0:
            fn_file = OUTPUT_TSV.replace('.tsv', '_false_negative.csv')
            false_neg.nsmallest(20, 'yes_prob')[['Prompt', 'full_image_path', 'yes_prob', 'yes_logit', 'no_logit']].to_csv(fn_file, index=False)
            print(f"✓ 已保存 {min(20, len(false_neg))} 个假阴性样本 → {fn_file}")

        if len(false_pos) > 0:
            fp_file = OUTPUT_TSV.replace('.tsv', '_false_positive.csv')
            false_pos.nlargest(20, 'yes_prob')[['Prompt', 'LocalPath', 'yes_prob', 'yes_logit', 'no_logit']].to_csv(fp_file, index=False)
            print(f"✓ 已保存 {min(20, len(false_pos))} 个假阳性样本 → {fp_file}")

        print("=" * 60)

        final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
        for i in range(accelerator.num_processes):
            os.remove(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"))


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 Qwen3_vl_reward_lora.py
