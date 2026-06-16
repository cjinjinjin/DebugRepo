import type { ChapterStepProps } from "../../registry/types";
import "./FlowOverview.css";

const gates = ["Local", "Parallel Prep", "Polaris", "Production", "Ops"];

export default function FlowOverview({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="fo-scene scene-pad">
        <div className="fo-kicker label-mono">DLIS DEPLOYMENT</div>
        <h1 className="fo-title">A good model can still fail at delivery.</h1>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="fo-scene scene-pad">
        <h2 className="fo-head">The five-gate spine</h2>
        <div className="fo-grid">
          {gates.map((gate, index) => (
            <div key={gate} className={`fo-card card ${index < 2 ? "fo-on" : ""}`}>
              <div className="label-mono">Gate {index + 1}</div>
              <div>{gate}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="fo-scene scene-pad">
        <h2 className="fo-head">Parallel lane by design</h2>
        <div className="fo-lanes">
          <div className="fo-lane card">
            <div className="label-mono">Lane A</div>
            <div>Checkpoint upload + Gen1 to Gen2 migration</div>
          </div>
          <div className="fo-lane card">
            <div className="label-mono">Lane B</div>
            <div>Branch push + CI image build</div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="fo-scene scene-pad">
        <h2 className="fo-head">Why v5.5 exists</h2>
        <div className="fo-tags">
          {["Gemma4", "ZImage", "ChangXu v2", "Hao doc", "Team incidents"].map((tag) => (
            <div key={tag} className="fo-tag card">
              {tag}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 4) {
    return (
      <div className="fo-scene scene-pad">
        <div className="fo-rule card">
          <div className="label-mono">NON-NEGOTIABLE</div>
          <div className="fo-rule-text">Local validation first.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="fo-scene scene-pad">
      <div className="fo-final card">
        <div className="hero-num">5</div>
        <div>All gates must pass before launch.</div>
      </div>
    </div>
  );
}
