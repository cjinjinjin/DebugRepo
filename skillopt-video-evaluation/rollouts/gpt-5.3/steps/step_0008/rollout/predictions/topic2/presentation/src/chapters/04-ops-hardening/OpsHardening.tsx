import type { ChapterStepProps } from "../../registry/types";
import "./OpsHardening.css";

export default function OpsHardening({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="oh-scene scene-pad">
        <h2 className="oh-head">SI and Prod must stay separated</h2>
        <div className="oh-split">
          <div className="oh-card card">SI endpoint</div>
          <div className="oh-card card">Prod endpoint</div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="oh-scene scene-pad">
        <h2 className="oh-head">Certificate lifecycle</h2>
        <div className="oh-time card">
          <span>issue</span>
          <span>rotation</span>
          <span>expiry alert</span>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="oh-scene scene-pad">
        <h2 className="oh-head">Kusto routing depends on cert context</h2>
        <div className="oh-route">
          <div className="oh-route-card card">SI cert → SI logs</div>
          <div className="oh-route-card card">Prod cert → Prod logs</div>
          <div className="oh-route-card card">Mismatch → wrong DB</div>
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="oh-scene scene-pad">
        <h2 className="oh-head">Logging anti-pattern fixes</h2>
        <div className="oh-fixes">
          {["no bare except pass", "use record.getMessage()", "flush on crash"].map((item) => (
            <div key={item} className="oh-fix card">
              {item}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 4) {
    return (
      <div className="oh-scene scene-pad">
        <h2 className="oh-head">Common issue heatmap</h2>
        <div className="oh-issues card">
          {["OOM fallback", "CUDA UUID", "read-only /Model", "timeout mirrors", "ACL mismatch"].map((x) => (
            <span key={x}>{x}</span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="oh-scene scene-pad">
      <h2 className="oh-head">Final readiness checklist</h2>
      <div className="oh-final card">
        {["image tag", "Gen2 integrity", "Polaris pass", "ACL", "endpoint URL", "cert expiry", "Kusto visible"].map((x) => (
          <span key={x}>{x}</span>
        ))}
      </div>
    </div>
  );
}
