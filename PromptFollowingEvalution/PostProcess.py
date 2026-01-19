import pandas as pd
import os, re

def post_process_refined(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return

    df = pd.read_csv(file_path, sep='\t', header=0)

    # 定义判定函数
    def determine_label(raw_text):
        text = str(raw_text).lower().replace('_', ' ').strip()
        no_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bwrong\b', 
            r'\bincorrect\b', r'\bfalse\b', r'\bnone\b'
        ]
        
        # 只要命中任何一个否定单词，就定性为 no
        if any(re.search(p, text) for p in no_patterns):
            return 'no'
        yes_patterns = [
            r'\byes\b', r'\babsolute\b', r'\babsolutely\b', 
            r'\bcorrect\b', r'\bright\b', r'\btrue\b'
        ]
        
        if any(re.search(p, text) for p in yes_patterns):
            return 'yes'
        
        return 'unknown'

    # 应用判定逻辑
    df['pred_answer'] = df['raw_output'].apply(determine_label)
    
    # 更新 is_yes 用于计数统计 (只有真正判为 yes 的才计 1)
    df['is_yes'] = (df['pred_answer'] == 'yes').astype(int)
    df['is_no'] = (df['pred_answer'] == 'no').astype(int)
    # 标记是否为有效回答 (yes 或 no)
    df['is_valid'] = df['pred_answer'].isin(['yes', 'no']).astype(int)

    # --- 聚合统计 ---
    summary_df = df.groupby(['UrlHash', 'Prompt']).agg(
        total_valid_questions=('is_valid', 'sum'),  # 只计入 yes 和 no 的行数
        yes_count=('is_yes', 'sum'),
        no_count=('is_no', 'sum'),
        actual_unknown_count=('pred_answer', lambda x: (x == 'unknown').sum()) # 仅作参考
    ).reset_index()

    summary_df['score'] = summary_df.apply(
        lambda x: x['yes_count'] / x['total_valid_questions'] if x['total_valid_questions'] > 0 else 0, 
        axis=1
    )

    # 保存结果
    output_detail = file_path.replace(".tsv", "_refined_v3.tsv")
    output_summary = file_path.replace(".tsv", "_refined_summary_v3.tsv")
    
    df.to_csv(output_detail, sep='\t', index=False)
    summary_df.to_csv(output_summary, sep='\t', index=False)
    
    print(f"处理完成！已保留 unknown 状态。")
    print(f"汇总文件包含 unknown_count 列，方便分析模型拒答情况。")

if __name__ == "__main__":
    target_file = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/super-realism-prompts_Qwen3_vl_output_results.tsv"
    post_process_refined(target_file)