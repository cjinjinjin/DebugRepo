"""
将 UHRS 标注的 CSV 数据转换为 ms-swift SFT 训练格式 (sharegpt JSON)

输入 CSV 列：
  - Prompt: 图片生成 prompt
  - LocalPath: 图片相对路径（需加 image_folder 前缀）
  - image4-promptFollowing: Good / fair / Bad

输出格式（ms-swift sharegpt）：
  [
    {
      "messages": [
        {"role": "user", "content": "<image>Does this image match ...? Answer Yes, Partial, or No."},
        {"role": "assistant", "content": "Yes"}
      ],
      "images": ["/absolute/path/to/image.jpg"]
    },
    ...
  ]
"""

import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================
DEFAULT_INPUT_CSV = (
    "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/"
    "AIGC/Data/AutoGen/PromptFollowing/"
    "UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
)
DEFAULT_IMAGE_FOLDER = (
    "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/"
    "AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
)
DEFAULT_OUTPUT_DIR = (
    "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/"
    "AIGC/Data/AutoGen/PromptFollowing/sft_data/"
)

LABEL_COL = "image4-promptFollowing"

# 标签 → 模型回答 映射
LABEL_MAP = {
    "Good": "Yes",
    "good": "Yes",
    "GOOD": "Yes",
    "fair": "Partial",
    "Fair": "Partial",
    "FAIR": "Partial",
    "Bad": "No",
    "bad": "No",
    "BAD": "No",
}

# Prompt 模板（与 Qwen3_vl_reward.py 保持一致风格）
PROMPT_TEMPLATE = (
    'Does this image match the description: "{prompt}"?\n'
    "Consider: Are the key objects present? Is the scene/composition correct? Are visual details accurate?\n"
    'Answer only "Yes", "Partial", or "No".'
)


def build_conversation(prompt: str, image_path: str, answer: str) -> dict:
    """构建单条 sharegpt 格式样本"""
    user_text = PROMPT_TEMPLATE.format(prompt=prompt)
    return {
        "messages": [
            {"role": "user", "content": f"<image>{user_text}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_path],
    }


def main(args):
    # ---------- 读取数据 ----------
    # 自动探测分隔符：先用 tab 试，不行再用逗号，最后用 python engine 兜底
    for sep, engine in [("\t", "c"), (",", "c"), (",", "python")]:
        try:
            df = pd.read_csv(
                args.input_csv,
                sep=sep,
                engine=engine,
                on_bad_lines="warn" if engine == "python" else "error",
            )
            if len(df.columns) > 1:
                print(f"读取成功 (sep={'TAB' if sep == chr(9) else sep}, engine={engine})")
                print(f"列名: {list(df.columns)}")
                break
        except Exception:
            continue
    else:
        raise RuntimeError("无法解析输入文件，请检查分隔符或文件格式")
    print(f"原始数据行数: {len(df)}")

    # ---------- 过滤 ----------
    # 1. 只保留有效标签
    df = df[df[LABEL_COL].isin(LABEL_MAP.keys())].copy()
    print(f"有效标签行数: {len(df)}")

    # 2. 过滤不存在的图片
    df["full_image_path"] = df["LocalPath"].apply(
        lambda x: os.path.join(args.image_folder, x)
    )
    before = len(df)
    df = df[df["full_image_path"].apply(os.path.exists)].copy()
    print(f"图片存在行数: {len(df)}  (过滤掉 {before - len(df)} 条缺失图片)")

    # ---------- 标签分布 ----------
    df["answer"] = df[LABEL_COL].map(LABEL_MAP)
    print("\n标签分布:")
    print(df["answer"].value_counts().to_string())

    # ---------- 构建对话样本 ----------
    records = []
    for _, row in df.iterrows():
        record = build_conversation(
            prompt=row["Prompt"],
            image_path=row["full_image_path"],
            answer=row["answer"],
        )
        records.append(record)

    print(f"\n总样本数: {len(records)}")

    # ---------- 划分 train / val ----------
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_ratio,
        random_state=42,
        # 按 answer 分层
        stratify=[r["messages"][1]["content"] for r in records],
    )
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # ---------- 保存 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"\n已保存:")
    print(f"  Train -> {train_path}")
    print(f"  Val   -> {val_path}")

    # ---------- 打印 1 条示例 ----------
    print("\n=== 示例样本 ===")
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 Qwen3-VL SFT 训练数据")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--image_folder", default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--val_ratio", type=float, default=0.05, help="验证集比例 (default: 0.05)"
    )
    args = parser.parse_args()
    main(args)
