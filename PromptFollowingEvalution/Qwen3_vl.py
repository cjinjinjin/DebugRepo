import os
import pandas as pd
import torch
import re
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

# 初始化 Accelerator
# 它会自动检测你的环境（单卡、多卡、TP、FSDP等）
accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"  # 确保路径正确
INPUT_TSV = "test_input.tsv"
OUTPUT_TSV = "output_results.tsv"
BATCH_SIZE = 4  # A100 上可以根据显存继续调大

# ================= 加载模型 =================
# 使用 accelerator.device 确保模型加载到正确的卡上
print(f"[{accelerator.process_index}] 正在加载模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # 当配合 accelerator 使用时，通常设为 None 或让 accelerator 处理
    device_map={"": accelerator.process_index}, 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 准备模型（在推理模式下，这主要处理设备移动）
model = accelerator.prepare(model)

def get_model_output(messages):
    """标准化的推理封装"""
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(accelerator.device) # 关键：使用 accelerator 的设备

    # 使用 accelerator.unwrap_model 获取原始模型进行 generate
    generated_ids = accelerator.unwrap_model(model).generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

# ================= 业务逻辑 =================
def main():
    if accelerator.is_main_process:
        df = pd.read_csv(INPUT_TSV, sep='\t')
    else:
        df = None
    
    # 简单的任务分发：这里演示逻辑，实际大规模建议用 DataLoader
    # 为了保证演示简单，我们假设在主进程处理，如果多卡，则需 split df
    # 如果你只有单块 A100，accelerator 依然能帮你处理掉 DTensor 的环境异常
    
    results = []
    
    # 模拟处理过程
    raw_data = pd.read_csv(INPUT_TSV, sep='\t')
    
    for _, row in tqdm(raw_data.iterrows(), total=len(raw_data), disable=not accelerator.is_main_process):
        # 1. 生成问题
        q_gen_msg = [[{"role": "user", "content": f"基于描述生成12个Yes/No问题：{row['prompt']}"}]]
        q_raw = get_model_output(q_gen_msg)[0]
        questions = [q.strip() for q in re.split(r'\d+\.', q_raw) if q.strip()][:12]
        
        # 2. 回答问题 (Batch 处理单图的12个问题)
        vqa_msgs = [[{
            "role": "user", 
            "content": [{"image": row['Imagepath']}, {"text": f"Question: {q}\nAnswer 'Yes' or 'No'."}]
        }] for q in questions]
        
        answers = get_model_output(vqa_msgs)
        
        # 统计
        yes_count = sum(1 for a in answers if "yes" in a.lower())
        no_count = sum(1 for a in answers if "no" in a.lower())
        qa_pairs = [f"Q: {q} | A: {a}" for q, a in zip(questions, answers)]
        
        results.append({
            **row.to_dict(),
            "qa_results": " || ".join(qa_pairs),
            "yes_count": yes_count,
            "no_count": no_count
        })

    if accelerator.is_main_process:
        pd.DataFrame(results).to_csv(OUTPUT_TSV, sep='\t', index=False)
        print(f"Done! Saved to {OUTPUT_TSV}")

if __name__ == "__main__":
    main()