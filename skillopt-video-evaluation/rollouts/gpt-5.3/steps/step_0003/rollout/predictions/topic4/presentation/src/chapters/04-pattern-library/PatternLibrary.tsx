import type { ChapterStepProps } from "../../registry/types";
import "./PatternLibrary.css";

const patternRows = [
  { p: "Daily Triage", c: "1d–2h", r: "Low", cost: "Low" },
  { p: "PR Babysitter", c: "5–15m", r: "Medium", cost: "High" },
  { p: "CI Sweeper", c: "5–15m", r: "High", cost: "Very high" },
  { p: "Issue Triage", c: "2h–1d", r: "Low", cost: "Low" },
];

export default function PatternLibraryChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="pl-scene scene-pad">
        <div className="kicker">Pattern catalog</div>
        <div className="pl-cards">
          {patternRows.map((r) => (
            <div key={r.p} className="card pl-pattern">{r.p}</div>
          ))}
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="pl-scene scene-pad">
        <div className="kicker">Cadence comparison</div>
        <div className="pl-cadence card">
          <div><span>Daily Triage</span><em style={{ width: "28%" }} /></div>
          <div><span>PR Babysitter</span><em style={{ width: "84%" }} /></div>
          <div><span>CI Sweeper</span><em style={{ width: "92%" }} /></div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="pl-scene scene-pad">
        <div className="kicker">Cost / risk map</div>
        <div className="pl-map card">
          <div className="pl-axis-x">Cost →</div>
          <div className="pl-axis-y">Risk →</div>
          <span className="pl-dot low">Daily</span>
          <span className="pl-dot mid">PR</span>
          <span className="pl-dot high">CI</span>
        </div>
      </div>
    );
  }
  if (step === 3) {
    return (
      <div className="pl-scene scene-pad">
        <div className="kicker">Pattern picker</div>
        <div className="pl-picker card">
          <div>Need fast confidence?</div>
          <div>Pick report-only L1 first</div>
          <div>Escalate autonomy only after audit pass</div>
        </div>
      </div>
    );
  }
  if (step === 4) {
    return (
      <div className="pl-scene scene-pad">
        <div className="kicker">Phased rollout</div>
        <div className="pl-phases">
          <div className="card">L1 Report</div>
          <div className="card">L2 Assisted</div>
          <div className="card">L3 Unattended</div>
        </div>
      </div>
    );
  }
  return (
    <div className="pl-scene scene-pad">
      <table className="pl-table card">
        <thead>
          <tr><th>Pattern</th><th>Cadence</th><th>Risk</th><th>Cost</th></tr>
        </thead>
        <tbody>
          {patternRows.map((r) => (
            <tr key={r.p}><td>{r.p}</td><td>{r.c}</td><td>{r.r}</td><td>{r.cost}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

