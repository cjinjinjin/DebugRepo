import type { ChapterStepProps } from "../../registry/types";
import "./LocalDataImage.css";

export default function LocalDataImage({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="ld-scene scene-pad">
        <div className="ld-branch card">
          <div className="label-mono">PERSONAL BRANCH</div>
          <div className="ld-path">&lt;alias&gt;/&lt;model-name&gt;</div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    const files = ["model.py", "dlis_inter.py", "http_server.py", "requirements-vllm.txt"];
    return (
      <div className="ld-scene scene-pad">
        <h2 className="ld-head">Core file map</h2>
        <div className="ld-grid">
          {files.map((f) => (
            <div key={f} className="ld-item card">
              <div className="label-mono">{f}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="ld-scene scene-pad">
        <h2 className="ld-head">Docker strategy</h2>
        <div className="ld-compare">
          <div className="ld-col card">
            <div className="label-mono">Dockerfile_vllm_fast</div>
            <div className="hero-num">&lt;1s</div>
          </div>
          <div className="ld-col card">
            <div className="label-mono">Pinned full stack</div>
            <div className="hero-num">~30m</div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="ld-scene scene-pad">
        <h2 className="ld-head">Local run loop</h2>
        <div className="ld-loop">
          {["build image", "run container", "send request", "check logs"].map((x, i) => (
            <div key={x} className="ld-step card">
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
      <div className="ld-scene scene-pad">
        <h2 className="ld-head">Data lane</h2>
        <div className="ld-tree card">
          <div className="label-mono">Gen1 flat layout → Gen2 migration</div>
          <div>root-level dlis_inter.py and cert files are required</div>
        </div>
      </div>
    );
  }

  return (
    <div className="ld-scene scene-pad">
      <h2 className="ld-head">Image lane</h2>
      <div className="ld-ci">
        <div className="ld-ci-card card">branch push triggers CI</div>
        <div className="ld-ci-card card">capture image tag output</div>
        <div className="ld-ci-card card">merge only when both lanes pass</div>
      </div>
    </div>
  );
}
