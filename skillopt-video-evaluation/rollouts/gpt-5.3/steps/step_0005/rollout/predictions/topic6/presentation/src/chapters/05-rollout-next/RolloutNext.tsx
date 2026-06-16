import type { ChapterStepProps } from "../../registry/types";
import "./RolloutNext.css";

export default function RolloutNextChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="t6a-scene scene-pad">
        <div className="t6a-kicker">ENV MATRIX</div>
        <div className="t6a-matrix">
          <div className="card t6a-item"><b>Codex</b><p>Mode B / 宿主执行</p></div>
          <div className="card t6a-item"><b>Claude Code / Cursor</b><p>Mode A / 自配 API</p></div>
          <div className="card t6a-item"><b>ChatGPT / Lovart</b><p>Mode C / 提示词顾问</p></div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="t6a-scene scene-pad">
        <div className="t6a-kicker">MODE A SETUP</div>
        <div className="t6a-terminal card">
          <pre>{`ENABLE_GARDEN_IMAGEGEN=true
OPENAI_BASE_URL=...
OPENAI_API_KEY=...`}</pre>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="t6a-scene scene-pad">
        <div className="t6a-kicker">ONE-LINE TO IMAGE</div>
        <div className="t6a-chain">
          <span>一句话需求</span>
          <span>结构化 prompt</span>
          <span>模型生成</span>
          <span>可交付图片</span>
        </div>
      </div>
    );
  }

  return (
    <div className="t6a-scene scene-pad">
      <div className="t6a-kicker">NEXT ACTION</div>
      <h2 className="t6a-title serif-cn">先抄近路，再做系统</h2>
      <ol className="t6a-list">
        <li>先去案例站，挑最接近的模板</li>
        <li>把 garden-skills 接进你的 Agent</li>
        <li>每次实战都把 prompt 回灌到模板里迭代</li>
      </ol>
    </div>
  );
}
