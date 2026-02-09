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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/" 
INPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_Qwen3_vl_8B_logits.tsv"
image_folder = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
BATCH_SIZE = 16  # 注意：Logits 模式显存占用略高于 Generate，如 OOM 可调小

# ================= 加载模型 =================
if accelerator.is_main_process:
    print(f"正在加载模型: {MODEL_PATH} ...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": accelerator.process_index}, 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"
# 获取 Yes 和 No 的 Token ID
# 注意：Qwen 的处理通常会区分大小写。根据 Prompt "Answer only 'Yes' or 'No'"，我们取首字母大写的 ID
yes_tokens = ["Yes", " Yes", "YES", "yes"]
no_tokens = ["No", " No", "NO", "no", "Part", "Partial", "Almost"] # 把半对半错的也归为 No 侧

ids_yes = [processor.tokenizer.convert_tokens_to_ids(t) for t in yes_tokens]
ids_no = [processor.tokenizer.convert_tokens_to_ids(t) for t in no_tokens]
YES_IDS = [i for i in ids_yes if i is not None]
NO_IDS = [i for i in ids_no if i is not None]
if accelerator.is_main_process:
    print(f"监控的 Yes Token IDs: {YES_IDS}")
    print(f"监控的 No Token IDs: {NO_IDS}")

model = accelerator.prepare(model)

# ================= 逻辑函数 =================

@torch.no_grad()
def run_inference_logits(msgs_batch):
    """
    通过 Forward 提取 Yes/No 的概率
    """
    inputs = processor.apply_chat_template(
        msgs_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True 
    ).to(accelerator.device)

    # 1. 正向传播获取 Logits
    outputs = model(**inputs)
    
    # 2. 获取最后一个位置（即预测首个回答 Token 的位置）的 Logits
    # outputs.logits 形状: [batch, sequence_length, vocab_size]
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    print(f"Actual generated: {decoded}")

    last_token_logits = outputs.logits[:, -1, :] 
    
    # ===== 新增：调试实际最常出现的token =====
    if not hasattr(run_inference_logits, "debug_count"):
        run_inference_logits.debug_count = 0
        run_inference_logits.top_tokens_collection = []
    
    if run_inference_logits.debug_count < 100 and accelerator.is_main_process:
        # 获取batch中每个样本的top-5 tokens
        topk_vals, topk_indices = torch.topk(last_token_logits, 5, dim=-1)
        for idx in range(topk_indices.shape[0]):
            if run_inference_logits.debug_count >= 100:
                break
            tokens = topk_indices[idx].cpu().tolist()
            decoded = processor.tokenizer.convert_ids_to_tokens(tokens)
            run_inference_logits.top_tokens_collection.append(decoded[0])  # 收集top-1
            
            if run_inference_logits.debug_count < 5:  # 只打印前5个详细信息
                print(f"\n[Sample {run_inference_logits.debug_count}] Top-5 tokens: {decoded}")
                print(f"  Logits: {topk_vals[idx].float().cpu().numpy()}")
            
            run_inference_logits.debug_count += 1
    
    # 在收集完100个样本后打印统计
    if run_inference_logits.debug_count == 100 and accelerator.is_main_process:
        from collections import Counter
        token_freq = Counter(run_inference_logits.top_tokens_collection)
        print("\n" + "="*50)
        print("前100个样本的Top-1 Token频率统计:")
        for token, count in token_freq.most_common(10):
            print(f"  '{token}': {count}次 ({count/100*100:.1f}%)")
        print("="*50 + "\n")
        run_inference_logits.debug_count += 1  # 防止重复打印

    # 3. 提取 Yes 和 No 的原始分值
    yes_logits = last_token_logits[:, YES_IDS].max(dim=-1).values
    no_logits = last_token_logits[:, NO_IDS].max(dim=-1).values
    
    # 4. 对这两者进行 Softmax，得到相对概率
    # 这样 yes_prob + no_prob = 1
    combined_logits = torch.stack([yes_logits, no_logits], dim=-1)
    probs = torch.softmax(combined_logits, dim=-1)
    
    # 返回 Yes 的概率作为 Score
    return probs[:, 0].float().cpu().numpy(), yes_logits.float().cpu().numpy(), no_logits.float().cpu().numpy()

# ================= 主程序 =================
def main():
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t', header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    my_df['FullLocalPath'] = my_df['LocalPath'].apply(lambda x: os.path.join(image_folder, x))
    
    # 过滤掉不存在的文件
    my_df = my_df[my_df['FullLocalPath'].apply(os.path.exists)].copy()
    if accelerator.is_main_process:
        print(f"模式: Logits 提取 (Prob of 'Yes')")
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    final_results = []
    buffer = []

    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        user_prompt = row['Prompt']
        image_path = row['FullLocalPath']

        prompt = f"""Examine if the image accurately represents: "{user_prompt}".
        Check if all major elements are present and correctly depicted. Answer only "Yes" or "No"."""
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        buffer.append({"msg": msg, "row_data": row.to_dict()})

        if len(buffer) == BATCH_SIZE:
            msgs_to_run = [b["msg"] for b in buffer]
            try:
                yes_probs, y_logits, n_logits = run_inference_logits(msgs_to_run)
                for idx, prob in enumerate(yes_probs):
                    res = buffer[idx]["row_data"]
                    res["yes_prob"] = float(prob)
                    res["yes_logit"] = float(y_logits[idx])
                    res["no_logit"] = float(n_logits[idx])
                    # 传统的分类判定：概率 > 0.5 即为 Yes
                    res["pred_answer"] = "yes" if prob > 0.5 else "no"
                    final_results.append(res)
            except Exception as e:
                print(f"Batch 推理失败: {e}")
            buffer = []

    # 处理收尾
    if buffer:
        msgs_to_run = [b["msg"] for b in buffer]
        yes_probs, y_logits, n_logits = run_inference_logits(msgs_to_run)
        for idx, prob in enumerate(yes_probs):
            res = buffer[idx]["row_data"]
            res["yes_prob"] = float(prob)
            res["yes_logit"] = float(y_logits[idx])
            res["no_logit"] = float(n_logits[idx])
            res["pred_answer"] = "yes" if prob > 0.5 else "no"
            final_results.append(res)

    # --- 保存临时文件与合并逻辑 (保持不变，但更新统计逻辑) ---
    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_output_file, sep='\t', index=False)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n>>> 正在合并结果并计算 AUC 指标...")
        all_dfs = [pd.read_csv(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"), sep='\t') 
                   for i in range(accelerator.num_processes)]
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        print("\n=== 诊断信息 ===")
        print(f"Yes概率分布: min={final_df['yes_prob'].min():.3f}, "
            f"mean={final_df['yes_prob'].mean():.3f}, "
            f"max={final_df['yes_prob'].max():.3f}")
        
        good_probs = final_df[final_df['image4-promptFollowing']=='Good']['yes_prob']
        bad_probs = final_df[final_df['image4-promptFollowing']=='Bad']['yes_prob']
        
        print(f"Good样本yes_prob均值: {good_probs.mean():.3f}")
        print(f"Bad样本yes_prob均值: {bad_probs.mean():.3f}")
        print(f"两者差距: {good_probs.mean() - bad_probs.mean():.3f}")

        # --- 重点：计算 AUC ---
        # 过滤出 Good 和 Bad 的行用于评估
        eval_df = final_df[final_df['image4-promptFollowing'].isin(['Good', 'Bad'])].copy()
    
        if len(eval_df) > 0:
            # 映射标签：Good=1, Bad=0（已经没有fair了）
            y_true = eval_df['image4-promptFollowing'].map({'Good': 1, 'Bad': 0})
            y_score = eval_df['yes_prob']
            
            try:
                auc_value = roc_auc_score(y_true, y_score)
                print("-" * 50)
                print(f"评估完成！")
                print(f"总样本数: {len(final_df)}")
                print(f"  - Good: {len(final_df[final_df['image4-promptFollowing']=='Good'])}")
                print(f"  - Bad: {len(final_df[final_df['image4-promptFollowing']=='Bad'])}")
                print(f"  - Fair: {len(final_df[final_df['image4-promptFollowing']=='fair'])}")
                print(f"有效评估样本数 (Good+Bad): {len(eval_df)}")
                print(f"Qwen3-VL Reward AUC: {auc_value:.4f}")
                
                # 【新增】分布诊断
                good_probs = eval_df[eval_df['image4-promptFollowing']=='Good']['yes_prob']
                bad_probs = eval_df[eval_df['image4-promptFollowing']=='Bad']['yes_prob']
                print(f"\nGood样本 yes_prob: mean={good_probs.mean():.3f}, std={good_probs.std():.3f}")
                print(f"Bad样本 yes_prob: mean={bad_probs.mean():.3f}, std={bad_probs.std():.3f}")
                print(f"均值差距: {good_probs.mean() - bad_probs.mean():.3f}")
                print("-" * 50)
                
                final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
            except Exception as e:
                print(f"AUC 计算失败: {e}")
        else:
            print("警告：未在 image4-promptFollowing 列中找到 'Good' 或 'Bad' 标签，无法计算 AUC。")

        # 清理临时文件
        for i in range(accelerator.num_processes):
            os.remove(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")) 

if __name__ == "__main__":
    main()