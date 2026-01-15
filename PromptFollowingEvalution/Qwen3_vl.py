import os
import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.device_mesh import init_device_mesh

# ================= 环境配置 =================
# 强制屏蔽分布式干扰，防止出现之前的 LOCAL_RANK KeyError
os.environ["LOCAL_RANK"] = "0" 

MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"  # 确保路径正确
INPUT_TSV = "test_input.tsv"
OUTPUT_TSV = "output_results.tsv"
BATCH_SIZE = 4  # A100 (40G/80G) 可以适当调大，例如 4-8

# ================= 加载模型 =================
print(f"正在加载模型至 A100...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, # A100 必用 bfloat16，速度快且省显存
    device_map="auto", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

def batch_get_response(batch_messages):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    
    input_kwargs = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        input_kwargs["images"] = image_inputs
    if video_inputs:
        input_kwargs["videos"] = video_inputs

    inputs = processor(**input_kwargs).to(model.device)

    # --- 修复 DTensor 冲突的关键代码 ---
    # 检查模型是否使用了分布式张量并行 (TP)
    # 如果 self_attn 的权重是 DTensor，则需要同步输入
    first_layer_weight = model.model.layers[0].self_attn.q_proj.weight
    if hasattr(first_layer_weight, "device_mesh"):
        mesh = first_layer_weight.device_mesh
        # 将 input_ids 等转换为 DTensor 并进行全复制 (Replicate) 策略
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # 这种转换能让输入张量与模型权重在同一个分布式网格中交互
                inputs[k] = distribute_tensor(v, mesh, [Replicate()])
    # -----------------------------------

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
# ================= 主流程 =================
def main():
    df = pd.read_csv(INPUT_TSV, sep='\t')
    all_results = []

    # 1. 批量生成问题
    print(f"Step 1: 正在批量生成问题 (Batch Size: {BATCH_SIZE})...")
    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        prompts = batch_df['prompt'].tolist()
        
        # 构造批量请求
        batch_msgs = [
            [{"role": "user", "content": f"基于描述生成12个Yes/No问题：{p}"}] 
            for p in prompts
        ]
        responses = batch_get_response(batch_msgs)
        
        # 解析并存入临时列表
        for idx, resp in enumerate(responses):
            questions = [q.strip() for q in re.split(r'\d+\.', resp) if q.strip()][:12]
            while len(questions) < 12: questions.append("Is the image clear?")
            
            row = batch_df.iloc[idx].to_dict()
            row['generated_questions'] = questions
            all_results.append(row)

    # 2. 批量回答问题 (VQA)
    print(f"Step 2: 正在批量进行视觉问答...")
    final_output = []
    for item in all_results:
        img_path = item['Imagepath']
        questions = item['generated_questions']
        
        ans_list = []
        yes_count, no_count = 0, 0
        
        # 这里对单张图的 12 个问题也可以做 batch
        vqa_msgs = [
            [{
                "role": "user",
                "content": [{"image": img_path}, {"text": f"Question: {q}\nAnswer only Yes or No."}]
            }] for q in questions
        ]
        
        # 将 12 个问题作为一个 batch 喂给显卡
        vqa_responses = batch_get_response(vqa_msgs)
        
        for q, a in zip(questions, vqa_responses):
            clean_a = "yes" if "yes" in a.lower() else ("no" if "no" in a.lower() else "unknown")
            if clean_a == "yes": yes_count += 1
            elif clean_a == "no": no_count += 1
            ans_list.append(f"Q: {q} | A: {clean_a}")
            
        item['qa_details'] = " || ".join(ans_list)
        item['yes_count'] = yes_count
        item['no_count'] = no_count
        final_output.append(item)

    # 保存
    pd.DataFrame(final_output).to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"全部任务完成！")

if __name__ == "__main__":
    main()