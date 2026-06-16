import type { ChapterStepProps } from "../../registry/types";
import "./AdoptionPlaybook.css";

export default function AdoptionPlaybookChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="ap-scene scene-pad">
        <div className="kicker">Step 1</div>
        <div className="ap-stage card">
          <h2>Single loop first</h2>
          <p>Bound one goal, one cadence, one owner.</p>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="ap-scene scene-pad">
        <div className="kicker">Step 2</div>
        <div className="ap-metrics card">
          <div><span>Outcomes</span><em>↑</em></div>
          <div><span>Incidents</span><em>↓</em></div>
          <div><span>Token drift</span><em>↓</em></div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="ap-scene scene-pad">
        <div className="kicker">Maturity path</div>
        <div className="ap-levels">
          <div className="card">L1<br />Report</div>
          <div className="card">L2<br />Assisted</div>
          <div className="card">L3<br />Unattended</div>
        </div>
      </div>
    );
  }
  return (
    <div className="ap-scene scene-pad ap-close">
      <div className="hero-num ap-num">07</div>
      <h2>Loop engineering = durable execution system</h2>
    </div>
  );
}

