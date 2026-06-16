import type { ChapterStepProps } from "../../registry/types";
import "./BlindtestLoop.css";

export default function BlindtestLoopChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="bl-scene scene-pad">
        <div className="kicker">SIDE BY SIDE</div>
        <div className="bl-arena">
          <div className="card bl-pane"><h3>候选 A</h3><p>隐藏模型名，仅展示回答质量。</p></div>
          <div className="card bl-pane"><h3>候选 B</h3><p>同题同条件流式输出对比。</p></div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="bl-scene scene-pad">
        <div className="kicker">VOTE</div>
        <div className="bl-votes card">
          <span>👈 左边更好</span><span>👉 右边更好</span><span>🤝 平局</span>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="bl-scene scene-pad">
        <div className="kicker">PROMPT CONFIG</div>
        <div className="bl-config card">
          <div><b>题目生成提示词</b><p>按领域约束生成题目</p></div>
          <div><b>作答提示词</b><p>统一模型答题上下文</p></div>
          <div><b>评分提示词</b><p>覆盖 scoreAnchors 细则</p></div>
        </div>
      </div>
    );
  }
  return (
    <div className="bl-scene scene-pad">
      <div className="kicker">CLOSED LOOP</div>
      <div className="bl-loop card">
        <div>测试集</div><div>自动评估</div><div>人工盲测</div><div>Prompt 优化</div><div>再次评估</div>
      </div>
      <p className="bl-end">把评估产品化，SFT 与 RAG 调优效率才可能真正翻倍。</p>
    </div>
  );
}
