import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/" 
INPUT_TSV = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/super-realism-prompts-new_withId_quetions.tsv"  # 格式: UrlHash \t Prompt \t Question
OUTPUT_TSV = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/super-realism-prompts_Qwen3_vl_output_results.tsv"
BATCH_SIZE = 12  # 每张显卡同时处理的问题数量，可根据显存调整

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

def run_inference(msgs_batch):
    """
    msgs_batch: List[List[Dict]]
    """
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
        max_new_tokens=10, # VQA只需回答Yes/No
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
        # 读取输入，显式指定分隔符为 tab
        full_df = pd.read_csv(INPUT_TSV, sep='\t', names=['UrlHash', 'Prompt', 'Question'], header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    # 数据分片：多卡并行处理不同行
    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    
    if accelerator.is_main_process:
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")

    final_results = []
    buffer = []

    # 遍历当前进程分配到的所有行
    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        url_hash = row['UrlHash']
        prompt = row['Prompt']
        question = row['Question']
        image_path = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/ZImage/Official/super-realism-prompts1k_official_20260116-0314/" + url_hash + "ZImage_w1344_h768.png"
        if pd.isna(question) or str(question).strip() == "":
            continue

        # 构造单条 VQA 消息
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, # 假设UrlHash是图片路径
                    {"type": "text", "text": f"Question: {question}\nAnswer only 'Yes' or 'No'."},
                ],
            }
        ]
        
        buffer.append({
            "msg": msg,
            "row_data": row.to_dict()
        })

        # 当 Buffer 达到 BATCH_SIZE 或者到数据末尾时，执行推理
        if len(buffer) == BATCH_SIZE:
            msgs_to_run = [b["msg"] for b in buffer]
            try:
                batch_answers = run_inference(msgs_to_run)
                for idx, ans in enumerate(batch_answers):
                    clean_ans = "yes" if "yes" in ans.lower() else ("no" if "no" in ans.lower() else "unknown")
                    # 合并原始数据和预测结果
                    res = buffer[idx]["row_data"]
                    res["pred_answer"] = clean_ans
                    res["raw_output"] = ans.strip()
                    final_results.append(res)
            except Exception as e:
                print(f"Batch 推理失败: {e}")
                for b in buffer:
                    res = b["row_data"]
                    res["pred_answer"] = "error"
                    res["raw_output"] = "error"
            
            buffer = [] # 清空缓冲

    # 处理最后剩余的 buffer
    if buffer:
        msgs_to_run = [b["msg"] for b in buffer]
        batch_answers = run_inference(msgs_to_run)
        for idx, ans in enumerate(batch_answers):
            clean_ans = "yes" if "yes" in ans.lower() else ("no" if "no" in ans.lower() else "unknown")
            res = buffer[idx]["row_data"]
            res["pred_answer"] = clean_ans
            res["raw_output"] = ans.strip()
            final_results.append(res)

    # --- 保存部分 ---
    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_output_file, sep='\t', index=False)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("正在合并结果并生成统计报表...")
        all_dfs = []
        for i in range(accelerator.num_processes):
            fname = OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")
            if os.path.exists(fname):
                all_dfs.append(pd.read_csv(fname, sep='\t'))
                os.remove(fname)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # 保存每一行对应一个 Question 的明细结果
            final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
            
            # --- 统计逻辑 ---
            final_df['is_yes'] = (final_df['pred_answer'] == 'yes').astype(int)
            summary_df = final_df.groupby(['UrlHash', 'Prompt']).agg(
                total_questions=('Question', 'count'),
                yes_count=('is_yes', 'sum')
            ).reset_index()
            
            summary_df['score'] = summary_df['yes_count'] / summary_df['total_questions']
            
            summary_filename = OUTPUT_TSV.replace(".tsv", "_summary.tsv")
            summary_df.to_csv(summary_filename, sep='\t', index=False)
            print(f"统计完成！\n明细文件: {OUTPUT_TSV}\n统计文件: {summary_filename}")

if __name__ == "__main__":
    main()