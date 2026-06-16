import type { ChapterStepProps } from "../../registry/types";
import "./PolarisProduction.css";

export default function PolarisProduction({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="pp-scene scene-pad">
        <div className="pp-gate card">
          <div className="label-mono">STEP 5</div>
          <div className="pp-title">Polaris quality gate</div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="pp-scene scene-pad">
        <h2 className="pp-head">High-risk configuration fields</h2>
        <div className="pp-matrix">
          {["ModelPath", "ModelDataPath", "Env Vars", "Ready Timeout"].map((x) => (
            <div key={x} className="pp-cell card">
              {x}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="pp-scene scene-pad">
        <h2 className="pp-head">Pass signal</h2>
        <div className="pp-bar card">
          <div className="pp-fill" />
          <div className="pp-label">loading 100% + success</div>
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="pp-scene scene-pad">
        <h2 className="pp-head">Test dimensions</h2>
        <div className="pp-dims">
          {["output", "latency", "stability", "resource"].map((x, i) => (
            <div key={x} className="pp-dim card">
              <div className="hero-num">{i + 1}</div>
              <div>{x}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 4) {
    return (
      <div className="pp-scene scene-pad">
        <div className="pp-loop card">fix → rebuild → retest</div>
      </div>
    );
  }

  return (
    <div className="pp-scene scene-pad">
      <h2 className="pp-head">Production cutover checks</h2>
      <div className="pp-prod">
        <div className="pp-prod-card card">hardware by workload</div>
        <div className="pp-prod-card card">Key → Hardware → General → ACL</div>
        <div className="pp-prod-card card">endpoint + cert request validation</div>
      </div>
    </div>
  );
}
