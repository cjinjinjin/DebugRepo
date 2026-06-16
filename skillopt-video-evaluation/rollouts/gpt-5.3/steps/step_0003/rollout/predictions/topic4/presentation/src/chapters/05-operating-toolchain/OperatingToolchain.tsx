import type { ChapterStepProps } from "../../registry/types";
import "./OperatingToolchain.css";

const rows = [
  { cmd: "npx @cobusgreyling/loop-init ...", result: "scaffold starter" },
  { cmd: "npx @cobusgreyling/loop-cost ...", result: "estimate spend" },
  { cmd: "npx @cobusgreyling/loop-audit ...", result: "readiness score" },
];

export default function OperatingToolchainChapter({ step }: ChapterStepProps) {
  if (step < 3) {
    const row = rows[step];
    return (
      <div className="ot-scene scene-pad">
        <div className="kicker">Command surface</div>
        <div className="ot-console card">
          {rows.map((r, i) => (
            <div key={r.cmd} className={`ot-row ${i === step ? "is-on" : ""}`}>
              <code>{r.cmd}</code><span>{r.result}</span>
            </div>
          ))}
        </div>
        <p className="ot-focus">{row.cmd}</p>
      </div>
    );
  }
  if (step === 3) {
    return (
      <div className="ot-scene scene-pad">
        <div className="kicker">Maturity gain</div>
        <div className="ot-gain">
          <div className="card"><h3>Before</h3><p>Unscored, ad-hoc, risky</p></div>
          <div className="card"><h3>After</h3><p>Audited, budgeted, gated</p></div>
        </div>
      </div>
    );
  }
  return (
    <div className="ot-scene scene-pad">
      <div className="kicker">Governance signal</div>
      <div className="ot-metric card">
        <div><span>Audit readiness</span><strong>88</strong></div>
        <div><span>Cost confidence</span><strong>91</strong></div>
        <div><span>Gate compliance</span><strong>94</strong></div>
      </div>
    </div>
  );
}

