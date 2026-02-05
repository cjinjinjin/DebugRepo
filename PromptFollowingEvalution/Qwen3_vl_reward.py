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

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/" 
INPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/BLIP3o/preprocess/occupation_ToyCaption_quetions.tsv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/BLIP3o/preprocess/occupation_ToyCaption_quetions_Qwen3_vl_logits.tsv"
image_folder = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/ZImage/Official/occupation_ZImage_official_20260120-2248/"
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

# 获取 Yes 和 No 的 Token ID
# 注意：Qwen 的处理通常会区分大小写。根据 Prompt "Answer only 'Yes' or 'No'"，我们取首字母大写的 ID
YES_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("Yes")
NO_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("No")

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
    last_token_logits = outputs.logits[:, -1, :] 
    
    # 3. 提取 Yes 和 No 的原始分值
    yes_logits = last_token_logits[:, YES_TOKEN_ID].float()
    no_logits = last_token_logits[:, NO_TOKEN_ID].float()
    
    # 4. 对这两者进行 Softmax，得到相对概率
    # 这样 yes_prob + no_prob = 1
    combined_logits = torch.stack([yes_logits, no_logits], dim=-1)
    probs = torch.softmax(combined_logits, dim=-1)
    
    # 返回 Yes 的概率作为 Score
    return probs[:, 0].cpu().numpy(), yes_logits.cpu().numpy(), no_logits.cpu().numpy()

# ================= 主程序 =================
def main():
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t', names=['UrlHash', 'Prompt', 'Question'], header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    
    if accelerator.is_main_process:
        print(f"模式: Logits 提取 (Prob of 'Yes')")
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    final_results = []
    buffer = []

    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        url_hash = row['UrlHash'].split('/')[-1].split('.')[0]
        question = row['Question']
        image_path = f"{image_folder}/{url_hash}.png"
        
        if pd.isna(question) or str(question).strip() == "":
            continue

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Question: {question}\nAnswer only 'Yes' or 'No'."},
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
        print("\n>>> 正在合并结果并计算 Reward Metrics...")
        all_dfs = [pd.read_csv(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"), sep='\t') 
                   for i in range(accelerator.num_processes) if os.path.exists(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"))]
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
            
            # 生成汇总表时，score 就是 yes_prob 的平均值
            summary_df = final_df.groupby(['UrlHash', 'Prompt']).agg(
                avg_yes_prob=('yes_prob', 'mean'),
                question_count=('yes_prob', 'count')
            ).reset_index()
            
            summary_filename = OUTPUT_TSV.replace(".tsv", "_summary.tsv")
            summary_df.to_csv(summary_filename, sep='\t', index=False)
            print(f"任务完成！连续得分(Probability)已保存。")
            # 清理临时文件
            for i in range(accelerator.num_processes):
                os.remove(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"))

if __name__ == "__main__":
    main()