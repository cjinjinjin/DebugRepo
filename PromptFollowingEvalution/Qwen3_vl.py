import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

# ================= 配置区 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"  # 确保路径正确
INPUT_TSV = "input.tsv"
OUTPUT_TSV = "output_results.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 加载模型 =================
print(f"正在加载模型: {MODEL_PATH}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

def get_response(messages):
    """通用的模型调用函数"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def parse_questions(text):
    """从模型输出中提取12个问题"""
    # 匹配 1. 2. 这种格式或换行符分割
    lines = re.split(r'\d+\.', text)
    questions = [q.strip() for q in lines if q.strip()]
    return questions[:12] # 确保只取前12个

# ================= 主程序 =================
def main():
    # 1. 读取输入
    df = pd.read_csv(INPUT_TSV, sep='\t')
    results = []

    for index, row in df.iterrows():
        img_id = row['id']
        prompt = row['prompt']
        img_path = row['Imagepath']
        
        print(f"正在处理 ID: {img_id}...")

        # --- 第一次调用：基于 Prompt 生成 12 个问题 ---
        msg_gen_q = [
            {"role": "user", "content": f"基于以下描述，生成12个关于图像内容的单选题（答案只能是Yes或No）：\n描述：{prompt}\n请直接列出问题，不要说废话。"}
        ]
        questions_raw = get_response(msg_gen_q)
        questions = parse_questions(questions_raw)
        
        # 确保生成了足够的问题
        while len(questions) < 12:
            questions.append("Is there something relevant in the image?")

        # --- 第二次调用：加载图片并回答这 12 个问题 ---
        ans_list = []
        yes_count = 0
        no_count = 0
        
        for i, q in enumerate(questions):
            msg_vqa = [
                {
                    "role": "user",
                    "content": [
                        {"image": img_path},
                        {"text": f"Question: {q}\nPlease answer with only 'Yes' or 'No'."},
                    ],
                }
            ]
            answer = get_response(msg_vqa).strip().lower()
            
            # 统计 Yes/No
            clean_ans = "yes" if "yes" in answer else ("no" if "no" in answer else "unknown")
            if clean_ans == "yes": yes_count += 1
            elif clean_ans == "no": no_count += 1
            
            ans_list.append(f"Q{i+1}: {q} | A: {clean_ans}")

        # 2. 整理结果
        res_entry = {
            "id": img_id,
            "prompt": prompt,
            "Imagepath": img_path,
            "questions_and_answers": " || ".join(ans_list),
            "yes_count": yes_count,
            "no_count": no_count
        }
        results.append(res_entry)

    # 3. 保存输出
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"处理完成！结果已保存至 {OUTPUT_TSV}")

if __name__ == "__main__":
    main()

# pip install torch torchvision transformers accelerate pandas pillow qwen-vl-utils