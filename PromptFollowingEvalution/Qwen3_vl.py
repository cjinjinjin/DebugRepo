import os
import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

# 1. 初始化 Accelerator
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"  # 确保路径正确
INPUT_TSV = "test_input.tsv"
OUTPUT_TSV = "output_results.tsv"

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

# 包装模型
model = accelerator.prepare(model)

def run_inference(msgs_batch):
    """
    msgs_batch: List[List[Dict]]
    """
    # [修复 1] 添加 padding=True，否则 Batch 推理会因为长度不齐报错
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
        max_new_tokens=256,
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
    # [修复 2] 多卡数据切分逻辑
    # 策略：所有进程都读取 CSV，然后利用切片 df[rank::world_size] 拿到属于自己的那份数据
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t')
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    # 数据分片：例如有2张卡，卡0处理偶数行，卡1处理奇数行
    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes]
    
    if accelerator.is_main_process:
        print(f"总数据量: {len(full_df)}, 当前进程处理: {len(my_df)}")

    results = []
    
    # 遍历当前进程分配到的数据
    for _, row in tqdm(my_df.iterrows(), total=len(my_df), disable=not accelerator.is_main_process):
        img_id = row['id']
        prompt = row['prompt']
        img_path = row['Imagepath']

        # --- Step 1: 纯文本生成问题 ---
        text_msgs = [[{
            "role": "user",
            "content": [{"type": "text", "text": f"基于以下描述，生成12个关于图像内容的Yes/No单选题：\n描述：{prompt}\n要求：只需列出问题，序号格式为1. 2. ..."}]
        }]]
        
        try:
            q_raw = run_inference(text_msgs)[0]
            questions = [q.strip() for q in re.split(r'\d+\.', q_raw) if q.strip()][:12]
        except Exception as e:
            print(f"ID {img_id} 生成问题失败: {e}")
            questions = []

        while len(questions) < 12:
            questions.append("Is the image content clear?")

        # --- Step 2: 图像问答 ---
        # [优化] 如果显存紧张，可以将 sub_batch_size 改为 4 或 6
        sub_batch_size = 12 
        answers = []
        
        # 将 12 个问题切分成小批次（防止超大分辨率图片导致 OOM）
        for i in range(0, len(questions), sub_batch_size):
            batch_qs = questions[i : i + sub_batch_size]
            vqa_batch = []
            for q in batch_qs:
                vqa_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": f"Question: {q}\nAnswer only 'Yes' or 'No'."},
                        ],
                    }
                ])
            
            try:
                batch_answers = run_inference(vqa_batch)
                answers.extend(batch_answers)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"ID {img_id} OOM, skipping batch...")
                    answers.extend(["unknown"] * len(batch_qs))
                    torch.cuda.empty_cache()
                else:
                    raise e

        # 统计结果
        yes_count = 0
        no_count = 0
        qa_details = []
        for q, a in zip(questions, answers):
            clean_a = "yes" if "yes" in a.lower() else ("no" if "no" in a.lower() else "unknown")
            if clean_a == "yes": yes_count += 1
            elif clean_a == "no": no_count += 1
            qa_details.append(f"Q: {q} | A: {clean_a}")

        results.append({
            "id": img_id,
            "prompt": prompt,
            "Imagepath": img_path,
            "all_qa": " || ".join(qa_details),
            "yes_count": yes_count,
            "no_count": no_count
        })

    # [修复 3] 结果汇聚与保存
    # 每个进程保存一个临时文件，最后（可选）合并，这里演示最安全的单进程保存
    # 简单做法：每个 rank 保存为 output_rank_x.tsv
    process_output_file = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(results).to_csv(process_output_file, sep='\t', index=False)
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()

    # 由主进程合并文件
    if accelerator.is_main_process:
        print("正在合并结果...")
        all_dfs = []
        for i in range(accelerator.num_processes):
            fname = OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")
            if os.path.exists(fname):
                all_dfs.append(pd.read_csv(fname, sep='\t'))
                os.remove(fname) # 合并后删除临时文件
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
            print(f"最终结果已保存至: {OUTPUT_TSV}")

if __name__ == "__main__":
    main()