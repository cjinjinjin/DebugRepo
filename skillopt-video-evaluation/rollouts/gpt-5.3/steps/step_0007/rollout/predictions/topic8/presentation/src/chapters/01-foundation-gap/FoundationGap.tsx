import type { ChapterStepProps } from "../../registry/types";
import "./FoundationGap.css";

export default function FoundationGapChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="fg-scene scene-pad">
        <div className="kicker">VERTICAL EVAL CRISIS</div>
        <h1 className="fg-title serif-cn">训练不难，评估才是效率瓶颈</h1>
        <div className="fg-strip card">
          <span>SFT</span><span>RAG</span><span>上线决策</span>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="fg-scene scene-pad">
        <div className="kicker">GENERIC SCORE GAP</div>
        <div className="fg-chart card">
          <div className="fg-col"><div className="hero-num">91</div><p>通用榜单</p></div>
          <div className="fg-divider" />
          <div className="fg-col"><div className="hero-num">58</div><p>垂域真实任务</p></div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="fg-scene scene-pad">
        <div className="kicker">THREE TARGETS</div>
        <div className="fg-goals">
          <div className="card fg-goal"><b>选型验证</b><span>谁先上场</span></div>
          <div className="card fg-goal"><b>SFT 验证</b><span>是否真提升</span></div>
          <div className="card fg-goal"><b>RAG 诊断</b><span>检索还是生成问题</span></div>
        </div>
      </div>
    );
  }

  return (
    <div className="fg-scene scene-pad">
      <div className="kicker">EVIDENCE LOOP</div>
      <div className="fg-loop card">
        <div>测试集</div><div>→</div><div>自动评估</div><div>→</div><div>人工盲测</div><div>→</div><div>策略回写</div>
      </div>
    </div>
  );
}
