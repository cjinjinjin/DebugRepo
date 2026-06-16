import type { ChapterStepProps } from "../../registry/types";
import "./MethodsLadder.css";

export default function MethodsLadderChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="ml-scene scene-pad">
        <div className="kicker">METHOD STACK</div>
        <div className="ml-stack">
          <div className="card ml-item">确定性评估</div>
          <div className="card ml-item">文本相似度</div>
          <div className="card ml-item ml-accent">LLM as Judge</div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="ml-scene scene-pad">
        <div className="kicker">OBJECTIVE QUESTIONS</div>
        <div className="ml-grid">
          <div className="card ml-card"><b>判断题</b><span>True / False</span></div>
          <div className="card ml-card"><b>单选题</b><span>A / B / C / D</span></div>
          <div className="card ml-card"><b>多选题</b><span>[A, C]</span></div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="ml-scene scene-pad">
        <div className="kicker">SIMILARITY LIMIT</div>
        <div className="ml-compare card">
          <div><p>标准答案</p><h2>利率上调</h2></div>
          <div className="ml-eq">≈</div>
          <div><p>模型回答</p><h2>加息</h2></div>
          <div className="ml-note">词面低重合 ≠ 语义错误</div>
        </div>
      </div>
    );
  }
  return (
    <div className="ml-scene scene-pad">
      <div className="kicker">JUDGE RUBRIC</div>
      <div className="ml-rubric card">
        <div className="ml-row"><span>准确性</span><b>40%</b></div>
        <div className="ml-row"><span>完整性</span><b>30%</b></div>
        <div className="ml-row"><span>可解释性</span><b>20%</b></div>
        <div className="ml-row"><span>安全性</span><b>10%</b></div>
      </div>
    </div>
  );
}
