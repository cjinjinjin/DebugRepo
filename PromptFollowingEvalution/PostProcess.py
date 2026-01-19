import pandas as pd
import os

def post_process_from_raw(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return

    # 读取包含 raw_output 的明细结果
    df = pd.read_csv(file_path, sep='\t', header=0)

    if 'raw_output' not in df.columns:
        print("Error: 文件中不存在 'raw_output' 列，无法重新判定。")
        return

    # --- 核心逻辑：基于 raw_output 重新判定 ---
    # 定义肯定的关键词：包含 yes 或 absolute (忽略大小写)
    # 也可以根据需要加入 'correct', 'true' 等
    positive_pattern = r'yes|absolute|absolutely|correct|true'
    
    # 使用 str.contains 进行模糊匹配
    # na=False 表示如果 raw_output 为空，则判定为 False
    df['is_yes'] = df['raw_output'].astype(str).str.lower().str.contains(positive_pattern, na=False).astype(int)

    # 为了方便核对，我们可以更新 pred_answer 列
    # 如果匹配到关键词设为 yes，否则保持原来的（或者设为 no）
    def update_label(row):
        if row['is_yes'] == 1:
            return 'yes'
        return 'no'
    
    df['pred_answer'] = df.apply(update_label, axis=1)

    # --- 聚合统计 ---
    summary_df = df.groupby(['UrlHash', 'Prompt']).agg(
        total_questions=('Question', 'count'),
        yes_count=('is_yes', 'sum')
    ).reset_index()

    # 计算得分
    summary_df['score'] = summary_df['yes_count'] / summary_df['total_questions']

    # 保存新的结果
    detail_filename = file_path.replace(".tsv", "_v2_fixed.tsv")
    summary_filename = file_path.replace(".tsv", "_v2_summary.tsv")
    
    df.to_csv(detail_filename, sep='\t', index=False)
    summary_df.to_csv(summary_filename, sep='\t', index=False)
    
    print(f"处理完成！")
    print(f"修正后的明细已保存: {detail_filename}")
    print(f"新的汇总报表已保存: {summary_filename}")

if __name__ == "__main__":
    target_file = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/super-realism-prompts_Qwen3_vl_output_results.tsv"
    post_process_from_raw(target_file)