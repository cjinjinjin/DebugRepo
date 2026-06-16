import type { ChapterStepProps } from "../../registry/types";
import "./WhyNow.css";

export default function WhyNowChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="wn-scene scene-pad">
        <div className="kicker">Momentum</div>
        <div className="wn-quotes">
          <article className="card wn-quote">
            <h3>Peter Steinberger</h3>
            <p>“Design loops that prompt your agents.”</p>
          </article>
          <article className="card wn-quote">
            <h3>Boris Cherny</h3>
            <p>“My job is to write loops.”</p>
          </article>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="wn-scene scene-pad">
        <div className="kicker">Tool span</div>
        <div className="wn-tool-grid">
          <div className="card">Grok</div>
          <div className="card">Claude Code</div>
          <div className="card">Codex</div>
          <div className="card">Cursor</div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="wn-scene scene-pad">
        <div className="kicker">Leverage shift</div>
        <div className="wn-bars card">
          <div><span>Prompt wording</span><em style={{ width: "24%" }} /></div>
          <div><span>Loop control</span><em style={{ width: "88%" }} /></div>
        </div>
      </div>
    );
  }

  return (
    <div className="wn-scene scene-pad wn-close">
      <div className="hero-num wn-num">02</div>
      <p>Why now is clear. Next: the concrete primitives.</p>
    </div>
  );
}

