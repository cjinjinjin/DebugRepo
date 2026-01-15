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
OUTPUT_TSV = "Qwen3_vl_output_results.tsv"

QuestionPrompt = """You are an expert Prompt Specification Analyst for Text-to-Image systems.\n\nYour task is NOT to evaluate an image.\nYour task is to deeply analyze a given IMAGE INPUT PROMPT\nand generate a set of verification questions that precisely\ncapture all explicit and implicit requirements in the prompt.\n\nThese questions will later be used to evaluate whether\na generated image truly follows the prompt.\n\nThere is NO image available at this stage.\nYou must reason ONLY from the prompt text itself.\n\n--------------------------------------------------\nOBJECTIVE\n--------------------------------------------------\n\nConvert the IMAGE INPUT PROMPT into a fixed-size set of\nYES / NO / N/A verification questions that reflect:\n\n- Text–Image Consistency (TIC)\n- Structural Integrity (SI)\n\nThe questions should expose:\n- hidden constraints\n- compositional requirements\n- relational logic\n- likely failure cases during generation\n\n--------------------------------------------------\nEVALUATION FRAMEWORK\n--------------------------------------------------\n\nYou MUST derive questions from the following dimensions\nONLY when they are explicitly stated or implicitly required\nby the prompt text.\n\n1. Linguistic Comprehension (TIC)\n- Negation (explicit absence or exclusion)\n- Attribute consistency (one attribute bound to multiple entities)\n- Co-reference resolution and ambiguity\n\n2. Visual Attributes (TIC)\n- Object-level counting (especially n ≥ 3)\n- Relative size or scale\n- Material properties\n- Facial expression or emotion (only if specified)\n- Artistic or visual style constraints\n\n3. Action & Interaction (TIC / SI)\n- Full-body actions\n- Hand or finger actions\n- Animal actions\n- Physical contact interactions\n- Non-contact interactions\n- Continuous states (e.g., flowing, floating, blowing)\n\n4. Relations & Structure (TIC)\n- Comparative relations\n- Compositional relations (entity made of other entities)\n- Containment relations\n- Shape similarity\n- Cross-entity attribute binding\n- Spatial layout or arrangement\n\n5. World Knowledge & Reasoning (TIC)\n- Named entities or landmarks\n- Knowledge-based identity claims\n- Counterfactual, surreal, or logically constrained conditions\n\n6. Scene Text & Typography (TIC)\n- Text content required to appear in-image\n- Text position, hierarchy, or layout constraints\n\n-------------------------------------------------\nMANDATORY COVERAGE OBLIGATIONS\n--------------------------------------------------\nDiscriminative Identity Obligation\nIf the prompt specifies a concrete entity, identity, role, or named concept,\nyou MUST generate at least one verification question\nthat confirms correct visual identification or classification of that entity,\nwithout relying on explicit negation.\n\nAttribute–Entity Binding Obligation\nFor any attribute, material, gear, logo, or abstract concept specified in the prompt,\nyou MUST generate at least one question\nverifying that the attribute is correctly bound\nto the intended entity and not detached, misplaced, or reassigned.\n\nSingular Entity Uniqueness Obligation\nIf the prompt uses an indefinite singular form (e.g., 'a', 'one')\nto introduce a primary entity,\nyou MUST generate one question verifying singularity of that entity.\nThis does not imply absence of other unrelated elements.\n\n--------------------------------------------------\nQUESTION GENERATION RULES\n--------------------------------------------------\n0. The absence of a constraint in the prompt MUST be treated as intentional freedom, not as an implicit requirement.\n1. Generate ONLY verification questions.\n2. Each question must test ONE specific requirement implied by the prompt.\n3. Questions must be answerable by inspecting a generated image.\n4. Use neutral, factual phrasing.\n5. Do NOT assume the image generator succeeds.\n   Questions should surface potential failure modes.\n6. If the prompt is vague or underspecified,\n   generate questions that explicitly expose this ambiguity.\n7. Do NOT invent requirements not grounded in the prompt text.\n8. You are evaluating the PROMPT ONLY.\n   Do NOT assume any image content exists.\n9. Do NOT generate any question that asserts the absence of an element, interaction, or property\nunless the IMAGE INPUT PROMPT explicitly requires such absence.\nHard Rule 1：No Purity Assumption\nDo NOT generate questions that assume visual purity, exclusivity, or 100% material consistency\nunless the prompt explicitly requires it (e.g., “entirely”, “pure”, “only”, “nothing but”).\nHard Rule 2：No Implicit Realism Constraint\nDo NOT generate questions that assume real-world scale, realism, or proportional correctness\nunless the prompt explicitly anchors scale or realism.\nHard Rule 3：No Aesthetic Optimization Questions\nDo NOT generate questions that evaluate compositional cleanliness, distraction, simplicity, or visual focus\nunless such criteria are explicitly stated in the prompt.\n\n1. No Default World Assumptions\nDo NOT assume realism, correct scale, natural proportions,\ntypical behavior, or physical plausibility unless explicitly stated.\n\n2. No Implicit Negation\nDo NOT generate negative or exclusionary questions\nunless the prompt explicitly specifies absence or prohibition.\n\n'Not mentioned' does NOT imply 'must not exist'.\n\n3. No Material Purity Expansion\nMaterial-related questions may only verify recognizability,\nNOT purity, uniformity, realism, or exclusion of other cues.\n\n4. No Background or Scene Cleanliness Constraints\nDo NOT impose requirements on background elements,\nbranding, text, objects, or scene composition\nunless explicitly required by the prompt.\n\n5. Every question must be ENTAILED by the prompt.\nIf a fully compliant image could answer 'no' to the question,\nthe question is INVALID and must not be generated.\n--------------------------------------------------\nOUTPUT FORMAT (STRICT)\n--------------------------------------------------\n\nYou MUST output exactly 12 questions.\n\nFormatting rules:\n- Each question MUST appear on its own line.\n- Each question MUST be enclosed in a unique XML-style tag:\n  <Q1>...</Q1>, <Q2>...</Q2>, ..., <Q12>...</Q12>\n- The questions MUST follow a fixed, deterministic order\n  aligned with the evaluation dimensions above.\n- Each question MUST be answerable with:\n  'yes', 'no', or 'n/a'.\n- Do NOT output explanations, headings, bullet points,\n  or any text outside the <Q*> tags.\n\n--------------------------------------------------\nINPUT\n--------------------------------------------------\n\nIMAGE INPUT PROMPT:\n{}\n--------------------------------------------------\nBEGIN\n\n"""
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
            "content": [{"type": "text", "text": QuestionPrompt.format(prompt)}]
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

# pip install git+https://github.com/huggingface/transformers    
# pip install flash-attn --no-build-isolation