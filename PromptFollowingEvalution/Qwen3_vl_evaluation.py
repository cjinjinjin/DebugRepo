import torch
if not hasattr(torch, "compiler"):
    class MockCompiler:
        def is_compiling(self): return False
    torch.compiler = MockCompiler()
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
import os, re
import pandas as pd

from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/" 
INPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/BLIP3o/preprocess/occupation_ToyCaption_quetions.tsv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/BLIP3o/preprocess/occupation_ToyCaption_quetions_Qwen3_vl_output_results.tsv"
image_folder = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/BLIP3o/occupation-NanoBanana/"
BATCH_SIZE = 48  

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
model = accelerator.prepare(model)

# ================= 逻辑函数 =================

def determine_label(raw_text):
    """
    判定逻辑：否定词优先，支持下划线替换，使用正则单词边界匹配
    """
    text = str(raw_text).lower().replace('_', ' ').strip()
    
    # 1. 判定否定 (Negative) - 优先级最高
    no_patterns = [
        r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bwrong\b', 
        r'\bincorrect\b', r'\bfalse\b', r'\bnone\b'
    ]
    if any(re.search(p, text) for p in no_patterns):
        return 'no'
    
    # 2. 判定肯定 (Positive)
    yes_patterns = [
        r'\byes\b', r'\babsolute\b', r'\babsolutely\b', 
        r'\bcorrect\b', r'\bright\b', r'\btrue\b'
    ]
    if any(re.search(p, text) for p in yes_patterns):
        return 'yes'
    
    return 'unknown'

def run_inference(msgs_batch):
    inputs = processor.apply_chat_template(
        msgs_batch,
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
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

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
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    final_results = []
    buffer = []

    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        url_hash = row['UrlHash'].split('/')[-1].split('.')[0]
        question = row['Question']
        # 修正路径拼接
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
                batch_answers = run_inference(msgs_to_run)
                for idx, ans in enumerate(batch_answers):
                    res = buffer[idx]["row_data"]
                    res["raw_output"] = ans.strip()
                    res["pred_answer"] = determine_label(ans)
                    final_results.append(res)
            except Exception as e:
                print(f"Batch 推理失败: {e}")
                for b in buffer:
                    res = b["row_data"]
                    res["raw_output"] = "error"
                    res["pred_answer"] = "unknown"
                    final_results.append(res)
            buffer = []

    if buffer:
        try:
            msgs_to_run = [b["msg"] for b in buffer]
            batch_answers = run_inference(msgs_to_run)
            for idx, ans in enumerate(batch_answers):
                res = buffer[idx]["row_data"]
                res["raw_output"] = ans.strip()
                res["pred_answer"] = determine_label(ans)
                final_results.append(res)
        except Exception as e:
            print(f"收尾 Batch 失败: {e}")

    # --- 保存临时文件 ---
    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_output_file, sep='\t', index=False)
    
    # ================= 结果合并与统计 (仅在主进程执行) =================
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n>>> 正在进行结果安全合并与报表生成...")
        all_dfs = []
        
        for i in range(accelerator.num_processes):
            fname = OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")
            
            if not os.path.exists(fname):
                print(f"警告: 找不到进程 {i} 的输出文件: {fname}")
                continue
                
            # 检查文件大小，如果是 0 字节则跳过
            if os.path.getsize(fname) == 0:
                print(f"跳过空文件 (0 bytes): {fname}")
                os.remove(fname)
                continue
                
            try:
                # 读取分片
                part_df = pd.read_csv(fname, sep='\t')
                if part_df is not None and not part_df.empty:
                    all_dfs.append(part_df)
                    print(f"成功读取 Rank {i}: {len(part_df)} 条记录")
                # 读取成功后删除临时文件
                os.remove(fname)
            except pd.errors.EmptyDataError:
                print(f"跳过无数据文件: {fname}")
                if os.path.exists(fname): os.remove(fname)
            except Exception as e:
                print(f"读取分片 {fname} 失败: {e}")

        # --- 开始统计 ---
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            # 1. 基础明细清洗
            final_df['is_yes'] = (final_df['pred_answer'] == 'yes').astype(int)
            final_df['is_no'] = (final_df['pred_answer'] == 'no').astype(int)
            final_df['is_valid'] = final_df['pred_answer'].isin(['yes', 'no']).astype(int)

            # 保存最终明细 TSV
            final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
            
            # 2. 聚合生成 Summary
            # 这里的 total_valid_questions 只统计 yes 和 no 的有效回答
            summary_df = final_df.groupby(['UrlHash', 'Prompt']).agg(
                total_valid_questions=('is_valid', 'sum'),
                yes_count=('is_yes', 'sum'),
                no_count=('is_no', 'sum'),
                unknown_count=('pred_answer', lambda x: (x == 'unknown').sum())
            ).reset_index()
            
            # 计算得分：yes / (yes + no)
            # 使用 numpy.where 或 fillna 处理分母为 0 的情况
            summary_df['score'] = 0.0
            mask = summary_df['total_valid_questions'] > 0
            summary_df.loc[mask, 'score'] = summary_df.loc[mask, 'yes_count'] / summary_df.loc[mask, 'total_valid_questions']
            
            summary_filename = OUTPUT_TSV.replace(".tsv", "_summary.tsv")
            summary_df.to_csv(summary_filename, sep='\t', index=False)
            
            print("-" * 50)
            print(f"任务圆满完成！")
            print(f"总计合并记录: {len(final_df)} 条")
            print(f"明细文件路径: {OUTPUT_TSV}")
            print(f"统计文件路径: {summary_filename}")
            print("-" * 50)
        else:
            print("错误：未能成功合并任何有效数据，请检查各进程推理日志。")
if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 Qwen3_vl_evaluation.py    