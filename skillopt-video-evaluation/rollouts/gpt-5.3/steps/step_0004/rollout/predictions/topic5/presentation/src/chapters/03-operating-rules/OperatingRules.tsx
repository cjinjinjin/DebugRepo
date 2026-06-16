import type { ChapterStepProps } from "../../registry/types";
import "./OperatingRules.css";

export default function OperatingRulesChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="or-scene scene-pad">
        <div className="kicker">ASK OR ACT</div>
        <div className="or-tree card">
          <div>信息充足</div><div className="or-arrow">→</div><div className="or-yes">直接执行</div>
          <div>语义缺失</div><div className="or-arrow">→</div><div className="or-no">补关键提问</div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="or-scene scene-pad">
        <div className="or-cases">
          <div className="card"><div className="label-mono">Case A</div><p>目标+时长都明确</p><b>NO ASK</b></div>
          <div className="card"><div className="label-mono">Case B</div><p>行为逻辑模糊</p><b>ASK</b></div>
          <div className="card"><div className="label-mono">Case C</div><p>输入上下文缺失</p><b>ASK</b></div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="or-scene scene-pad">
        <div className="kicker">EXTREMELY BRIEF SUMMARY</div>
        <div className="or-sum card">
          <div className="or-strike">我做了什么、怎么做的、每一步都复述一遍……</div>
          <div className="or-keep">Caveat: 还缺真实素材。</div>
          <div className="or-keep">Next step: 补图并跑最终录屏。</div>
        </div>
      </div>
    );
  }
  return (
    <div className="or-scene scene-pad">
      <div className="kicker">ONE THOUSAND NO'S</div>
      <div className="or-quota">
        <div className="or-ring"><span>1000 NO</span></div>
        <div className="or-plus">→</div>
        <div className="or-yes-block card">1 YES<br/>每个元素都要有理由</div>
      </div>
    </div>
  );
}
