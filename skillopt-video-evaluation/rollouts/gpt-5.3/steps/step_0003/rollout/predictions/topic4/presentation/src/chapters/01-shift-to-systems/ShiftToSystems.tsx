import type { ChapterStepProps } from "../../registry/types";
import "./ShiftToSystems.css";

export default function ShiftToSystemsChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="sts-scene scene-pad">
        <div className="masthead">
          <span className="brand">Loop Engineering</span>
          <span className="issue">Chapter 01</span>
        </div>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />
        <div className="sts-title-wrap">
          <div className="kicker">Old mode</div>
          <h1 className="sts-title">Human writes every prompt</h1>
          <div className="sts-terminal card">
            <div className="sts-row"><span>$</span> write prompt v1</div>
            <div className="sts-row"><span>$</span> tweak prompt v2</div>
            <div className="sts-row"><span>$</span> retry prompt v3</div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="sts-scene scene-pad">
        <div className="kicker">New mode</div>
        <div className="sts-split">
          <div className="hero-num sts-num">01</div>
          <div className="sts-flow card">
            <div>Goal</div>
            <div>Scheduler</div>
            <div>Agent Prompting Loop</div>
            <div>Verification</div>
          </div>
        </div>
        <p className="sts-lead">Leverage moves from phrasing prompts to designing control flow.</p>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="sts-scene scene-pad">
        <div className="kicker">Definition</div>
        <div className="sts-def card">
          <div className="sts-def-item"><b>Goal</b><span>clear purpose and stop condition</span></div>
          <div className="sts-def-item"><b>Cadence</b><span>scheduled and repeatable execution</span></div>
          <div className="sts-def-item"><b>Verification</b><span>tests, audits, and gates</span></div>
          <div className="sts-def-item"><b>Memory</b><span>durable state across runs</span></div>
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="sts-scene scene-pad">
        <div className="kicker">Cycle</div>
        <div className="sts-cycle">
          <div className="sts-node">Iterate</div>
          <div className="sts-arrow">→</div>
          <div className="sts-node">Verify</div>
          <div className="sts-arrow">→</div>
          <div className="sts-node">Handoff</div>
          <div className="sts-arrow">↺</div>
        </div>
      </div>
    );
  }

  return (
    <div className="sts-scene scene-pad sts-close">
      <div className="pull-quote">
        You are not writing prompts.
        <br />
        You are designing the prompting system.
      </div>
    </div>
  );
}

