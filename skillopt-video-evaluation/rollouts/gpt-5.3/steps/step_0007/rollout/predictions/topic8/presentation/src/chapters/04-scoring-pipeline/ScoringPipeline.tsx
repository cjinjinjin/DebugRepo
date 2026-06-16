import type { ChapterStepProps } from "../../registry/types";
import "./ScoringPipeline.css";

export default function ScoringPipelineChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="sp-scene scene-pad">
        <div className="kicker">AUTO GRADING</div>
        <div className="sp-dual card">
          <div><b>客观题</b><span>规则比对</span></div>
          <div className="sp-arrow">→</div>
          <div><b>主观题</b><span>教师模型评估</span></div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="sp-scene scene-pad">
        <div className="kicker">TEACHER JUDGE</div>
        <div className="sp-score card">
          <h2>7.8 / 10</h2>
          <p>理由：答案覆盖关键事实，但案例解释不完整。</p>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="sp-scene scene-pad">
        <div className="kicker">SCALE STRATEGY</div>
        <div className="sp-scale card">
          <div className="sp-stage"><b>20 题</b><span>校准评分标准</span></div>
          <div className="sp-stage"><b>200 题</b><span>扩大对比范围</span></div>
          <div className="sp-stage"><b>持续回归</b><span>版本跟踪</span></div>
        </div>
      </div>
    );
  }
  return (
    <div className="sp-scene scene-pad">
      <div className="kicker">EVAL DASHBOARD</div>
      <div className="sp-board">
        <div className="card sp-kpi"><span>准确率</span><b>84%</b></div>
        <div className="card sp-kpi"><span>幻觉率</span><b>6%</b></div>
        <div className="card sp-kpi"><span>主观得分</span><b>7.9</b></div>
      </div>
    </div>
  );
}
