import type { ChapterStepProps } from "../../registry/types";
import "./PromptCore.css";

export default function PromptCoreChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="pc-scene scene-pad">
        <div className="kicker">SYSTEM PROMPT</div>
        <div className="pc-center">
          <div className="pc-big hero-num">420</div>
          <div className="pc-cap">LINES OF HIGH-DENSITY INSTRUCTIONS</div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="pc-scene scene-pad">
        <div className="pc-row">
          <div className="card pc-card"><div className="label-mono">AI</div><h2>Designer</h2></div>
          <div className="pc-link">×</div>
          <div className="card pc-card"><div className="label-mono">User</div><h2>Manager</h2></div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="pc-scene scene-pad">
        <div className="kicker">DYNAMIC IDENTITIES</div>
        <div className="pc-wheel">
          <div className="card">Animator</div>
          <div className="card">UX Designer</div>
          <div className="card">Slide Designer</div>
          <div className="card">Prototyper</div>
        </div>
      </div>
    );
  }
  return (
    <div className="pc-scene scene-pad">
      <div className="pc-compare">
        <div className="pc-col card"><div className="label-mono">静态角色</div><p>你是前端开发者</p></div>
        <div className="pc-col card pc-yes"><div className="label-mono">动态角色</div><p>按任务切换专业身份</p></div>
      </div>
    </div>
  );
}
