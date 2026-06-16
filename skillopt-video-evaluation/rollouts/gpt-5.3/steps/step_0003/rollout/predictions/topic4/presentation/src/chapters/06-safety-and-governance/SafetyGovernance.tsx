import type { ChapterStepProps } from "../../registry/types";
import "./SafetyGovernance.css";

export default function SafetyGovernanceChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="sg-scene scene-pad">
        <div className="kicker">Failure map</div>
        <div className="sg-grid">
          <div className="card">Token blowups</div>
          <div className="card">Unchecked merges</div>
          <div className="card">Comprehension debt</div>
          <div className="card">Blind retries</div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="sg-scene scene-pad">
        <div className="kicker">Multi-loop collision</div>
        <div className="sg-collision card">
          <div className="sg-lane a">Loop A</div>
          <div className="sg-lane b">Loop B</div>
          <div className="sg-lane c">Shared target</div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="sg-scene scene-pad">
        <div className="kicker">Security constraints</div>
        <div className="sg-lock card">
          <div>Connector scope</div>
          <div>Denylist controls</div>
          <div>Unattended risk policy</div>
        </div>
      </div>
    );
  }
  if (step === 3) {
    return (
      <div className="sg-scene scene-pad">
        <div className="kicker">Human gate branch</div>
        <div className="sg-branch">
          <div className="card">Safe / allowlisted → Execute</div>
          <div className="card">Risky / ambiguous → Escalate</div>
        </div>
      </div>
    );
  }
  return (
    <div className="sg-scene scene-pad sg-close">
      <div className="pull-quote">Automate execution, not responsibility.</div>
    </div>
  );
}

