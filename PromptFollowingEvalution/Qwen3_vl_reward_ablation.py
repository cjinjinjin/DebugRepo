"""
Qwen3-VL Reward Model — Multi-Strategy Ablation
================================================
在同一次推理中跑 4 种打分策略，最后对比 AUC / Spearman / Fair分布。

策略列表：
  v1_yesno      : 原始 Yes/No logit（方案一基准）
  v2_grade      : A-E 序数等级 logit 加权（方案二）
  v3_yesno_cot  : 加入 CoT 引导的 Yes/No prompt
  v4_grade_cot  : 加入 CoT 引导的 A-E 序数 prompt

每张图仅需 4 次 forward，通过 STRATEGY 配置可以灵活增删。
"""

import torch
if not hasattr(torch, "compiler"):
    class MockCompiler:
        def is_compiling(self): return False
    torch.compiler = MockCompiler()
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

accelerator = Accelerator()

# ================= 配置 =================
MODEL_PATH = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct/"
INPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
OUTPUT_TSV = "/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_Qwen3_ablation.tsv"
IMAGE_FOLDER = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
BATCH_SIZE = 8   # 4 策略同批，显存翻倍，适当调小

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

# ================= Token ID 初始化 =================
yes_tokens   = ["Yes", " Yes", "yes", " yes", "YES", " YES", "Ġyes", "ĠYes"]
no_tokens    = ["No", " No", "no", " no", "NO", " NO", "Ġno", "ĠNo"]
partial_tokens = ["Part", " Part", "Partial", " Partial", "Almost", " Almost"]

YES_IDS     = [i for i in [processor.tokenizer.convert_tokens_to_ids(t) for t in yes_tokens]
               if i is not None and i >= 0]
NO_IDS      = [i for i in [processor.tokenizer.convert_tokens_to_ids(t) for t in no_tokens]
               if i is not None and i >= 0]
PARTIAL_IDS = [i for i in [processor.tokenizer.convert_tokens_to_ids(t) for t in partial_tokens]
               if i is not None and i >= 0]

grade_token_map = {
    "A": ["A", " A", "ĠA"],
    "B": ["B", " B", "ĠB"],
    "C": ["C", " C", "ĠC"],
    "D": ["D", " D", "ĠD"],
    "E": ["E", " E", "ĠE"],
}
GRADE_IDS = {
    g: [i for i in [processor.tokenizer.convert_tokens_to_ids(t) for t in toks]
        if i is not None and i >= 0]
    for g, toks in grade_token_map.items()
}
GRADE_WEIGHTS = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}

if accelerator.is_main_process:
    print(f"YES_IDS: {YES_IDS}")
    print(f"NO_IDS:  {NO_IDS}")
    for g, ids in GRADE_IDS.items():
        print(f"Grade {g} IDs: {ids}")

model = accelerator.prepare(model)


# ================= Prompt 构造 =================
def make_prompt_yesno(user_prompt: str) -> str:
    return (f'Does this image match the description: "{user_prompt}"?\n'
            f"Consider: Are the key objects present? Is the scene/composition correct? "
            f"Are visual details accurate?\n"
            f'Answer only "Yes" or "No".')


def make_prompt_grade(user_prompt: str) -> str:
    return (f'Rate how well this image matches the description: "{user_prompt}"\n\n'
            f"A: Excellent — all key elements match, composition and details are accurate\n"
            f"B: Good — most elements match with minor omissions or inaccuracies\n"
            f"C: Fair — some elements match but notable aspects are missing or wrong\n"
            f"D: Poor — few elements match, significant misalignment\n"
            f"E: Very Poor — does not match the description at all\n\n"
            f"Answer with a single letter (A, B, C, D, or E):")


def make_prompt_yesno_cot(user_prompt: str) -> str:
    return (f'Does this image match the description: "{user_prompt}"?\n\n'
            f"Step 1: List the main objects/elements visible in the image.\n"
            f"Step 2: Check if the description's key requirements are satisfied.\n"
            f"Step 3: Give a final judgment.\n\n"
            f'Final answer (Yes or No):')


def make_prompt_grade_cot(user_prompt: str) -> str:
    return (f'Rate how well this image matches the description: "{user_prompt}"\n\n'
            f"Step 1: List the main objects/elements visible in the image.\n"
            f"Step 2: Check each requirement in the description.\n"
            f"Step 3: Give an overall rating.\n\n"
            f"A: Excellent  B: Good  C: Fair  D: Poor  E: Very Poor\n\n"
            f"Final rating (A/B/C/D/E):")


STRATEGY_PROMPT_FNS = {
    "v1_yesno":     make_prompt_yesno,
    "v2_grade":     make_prompt_grade,
    "v3_yesno_cot": make_prompt_yesno_cot,
    "v4_grade_cot": make_prompt_grade_cot,
}


# ================= 打分函数 =================
def score_yesno(last_logits: torch.Tensor) -> np.ndarray:
    """last_logits: (batch, vocab)"""
    yes_l   = last_logits[:, YES_IDS].max(dim=-1).values
    no_l    = last_logits[:, NO_IDS].max(dim=-1).values
    part_l  = (last_logits[:, PARTIAL_IDS].max(dim=-1).values
               if len(PARTIAL_IDS) > 0 else torch.zeros_like(yes_l))
    stacked = torch.stack([no_l, part_l, yes_l], dim=-1)
    w = torch.tensor([0.0, 0.5, 1.0], device=last_logits.device)
    probs = torch.softmax(stacked, dim=-1)
    return (probs * w).sum(dim=-1).float().cpu().numpy()


def score_grade(last_logits: torch.Tensor) -> np.ndarray:
    """last_logits: (batch, vocab)"""
    grade_l = torch.stack(
        [last_logits[:, GRADE_IDS[g]].max(dim=-1).values for g in "ABCDE"],
        dim=-1
    )  # (batch, 5)
    probs = torch.softmax(grade_l, dim=-1)
    w = torch.tensor([GRADE_WEIGHTS[g] for g in "ABCDE"],
                     dtype=torch.float32, device=last_logits.device)
    return (probs * w).sum(dim=-1).float().cpu().numpy()


STRATEGY_SCORE_FNS = {
    "v1_yesno":     score_yesno,
    "v2_grade":     score_grade,
    "v3_yesno_cot": score_yesno,   # CoT 版用相同 token 打分
    "v4_grade_cot": score_grade,
}


# ================= 单策略 forward =================
@torch.no_grad()
def run_strategy(msgs_batch: list, score_fn) -> np.ndarray:
    inputs = processor.apply_chat_template(
        msgs_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(accelerator.device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    return score_fn(last_logits)


# ================= 主程序 =================
def main():
    try:
        full_df = pd.read_csv(INPUT_TSV, sep='\t', header=0)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"读取文件失败: {e}")
        return

    my_df = full_df.iloc[accelerator.process_index :: accelerator.num_processes].copy()
    my_df["FullLocalPath"] = my_df["LocalPath"].apply(
        lambda x: os.path.join(IMAGE_FOLDER, x)
    )
    my_df = my_df[my_df["FullLocalPath"].apply(os.path.exists)].copy()

    if accelerator.is_main_process:
        print(f"总行数: {len(full_df)}, 当前进程处理行数: {len(my_df)}")
        print(f"运行策略: {list(STRATEGY_PROMPT_FNS.keys())}")

    strategies = list(STRATEGY_PROMPT_FNS.keys())
    final_results = []
    buffer = []

    for _, row in tqdm(my_df.iterrows(), total=len(my_df),
                       disable=not accelerator.is_main_process):
        user_prompt = row["Prompt"]
        image_path  = row["FullLocalPath"]
        buffer.append({"user_prompt": user_prompt, "image_path": image_path,
                        "row_data": row.to_dict()})

        if len(buffer) == BATCH_SIZE:
            row_scores = {s: None for s in strategies}
            for strategy in strategies:
                prompt_fn = STRATEGY_PROMPT_FNS[strategy]
                score_fn  = STRATEGY_SCORE_FNS[strategy]
                msgs = [
                    [{"role": "user", "content": [
                        {"type": "image", "image": b["image_path"]},
                        {"type": "text",  "text":  prompt_fn(b["user_prompt"])},
                    ]}]
                    for b in buffer
                ]
                try:
                    row_scores[strategy] = run_strategy(msgs, score_fn)
                except Exception as e:
                    print(f"[{strategy}] Batch failed: {e}")
                    row_scores[strategy] = np.full(len(buffer), 0.5)

            for idx, b in enumerate(buffer):
                res = b["row_data"].copy()
                for s in strategies:
                    res[f"score_{s}"] = float(row_scores[s][idx])
                final_results.append(res)
            buffer = []

    # 处理收尾
    if buffer:
        row_scores = {s: None for s in strategies}
        for strategy in strategies:
            prompt_fn = STRATEGY_PROMPT_FNS[strategy]
            score_fn  = STRATEGY_SCORE_FNS[strategy]
            msgs = [
                [{"role": "user", "content": [
                    {"type": "image", "image": b["image_path"]},
                    {"type": "text",  "text":  prompt_fn(b["user_prompt"])},
                ]}]
                for b in buffer
            ]
            try:
                row_scores[strategy] = run_strategy(msgs, score_fn)
            except Exception as e:
                print(f"[{strategy}] Tail batch failed: {e}")
                row_scores[strategy] = np.full(len(buffer), 0.5)
        for idx, b in enumerate(buffer):
            res = b["row_data"].copy()
            for s in strategies:
                res[f"score_{s}"] = float(row_scores[s][idx])
            final_results.append(res)

    # 保存分片结果
    process_out = OUTPUT_TSV.replace(".tsv", f"_rank_{accelerator.process_index}.tsv")
    pd.DataFrame(final_results).to_csv(process_out, sep='\t', index=False)
    accelerator.wait_for_everyone()

    # ================= 主进程汇总与对比 =================
    if accelerator.is_main_process:
        all_dfs = [
            pd.read_csv(OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv"), sep='\t')
            for i in range(accelerator.num_processes)
        ]
        final_df = pd.concat(all_dfs, ignore_index=True)

        label_col = "image4-promptFollowing"
        binary_df = final_df[final_df[label_col].isin(["Good", "Bad"])].copy()
        y_bin     = binary_df[label_col].map({"Good": 1, "Bad": 0})

        ordinal_df = final_df[final_df[label_col].isin(["Good", "Bad", "fair"])].copy()
        y_ord      = ordinal_df[label_col].map({"Bad": 0, "fair": 1, "Good": 2})

        print("\n" + "=" * 70)
        print(f"{'Strategy':<18} {'AUC':>6}  {'Spearman':>9}  "
              f"{'GoodMean':>9}  {'FairMean':>9}  {'BadMean':>8}  "
              f"{'FairMid%':>9}")
        print("-" * 70)

        for s in strategies:
            col = f"score_{s}"
            auc = roc_auc_score(y_bin, binary_df[col])
            sp  = spearmanr(y_ord, ordinal_df[col]).statistic

            g_mean = final_df[final_df[label_col] == "Good"][col].mean()
            f_mean = final_df[final_df[label_col] == "fair"][col].mean()
            b_mean = final_df[final_df[label_col] == "Bad"][col].mean()

            fair_rows = final_df[final_df[label_col] == "fair"][col]
            fair_mid_pct = ((fair_rows >= 0.33) & (fair_rows <= 0.67)).mean() * 100

            print(f"{s:<18} {auc:>6.4f}  {sp:>9.4f}  "
                  f"{g_mean:>9.3f}  {f_mean:>9.3f}  {b_mean:>8.3f}  "
                  f"{fair_mid_pct:>8.1f}%")

        print("=" * 70)

        # --------- 各策略 Fair 样本分布明细 ---------
        print("\n=== Fair 样本分布明细 ===")
        for s in strategies:
            col = f"score_{s}"
            fair_rows = final_df[final_df[label_col] == "fair"][col]
            n = len(fair_rows)
            bad_pct  = (fair_rows < 0.33).mean() * 100
            mid_pct  = ((fair_rows >= 0.33) & (fair_rows <= 0.67)).mean() * 100
            good_pct = (fair_rows > 0.67).mean() * 100
            print(f"  {s:<18}  Bad<0.33: {bad_pct:5.1f}%  Mid: {mid_pct:5.1f}%  Good>0.67: {good_pct:5.1f}%  (n={n})")

        # 保存完整结果
        final_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
        print(f"\n完整结果已保存 → {OUTPUT_TSV}")

        # 清理分片文件
        for i in range(accelerator.num_processes):
            tmp = OUTPUT_TSV.replace(".tsv", f"_rank_{i}.tsv")
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    main()
