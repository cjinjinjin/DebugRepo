import type { ChapterStepProps } from "../../registry/types";
import "./ConceptDeploy.css";

const LAYERS = ["Inference", "Memory", "RAG", "MCP", "Skills"];

export default function ConceptDeploy({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="cd-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">02 / Concept</span></div>
        <div className="cd-stack">
          {LAYERS.map((layer, i) => <div className="cd-layer card" key={layer} style={{ animationDelay: `${i * 70}ms` }}>{layer}</div>)}
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="cd-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">02 / Concept</span></div>
        <div className="cd-memory">
          <div className="cd-box card"><h3>无状态问题</h3><p>推理服务处理完请求就结束，会话无法自然延续。</p></div>
          <div className="cd-box card"><h3>记忆机制</h3><p>短期保原文，长期做摘要与实体抽取，再在下一轮拼装。</p></div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="cd-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">02 / Deploy</span></div>
        <div className="cd-panel card">
          <div className="cd-kicker">Deployment Path</div>
          <div className="cd-line"><b>1</b> 环境校验（Node.js 20+）</div>
          <div className="cd-line"><b>2</b> 安装与初始化向导</div>
          <div className="cd-line"><b>3</b> 基础链路联调</div>
        </div>
      </div>
    );
  }

  return (
    <div className="cd-scene scene-pad">
      <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">02 / Deploy</span></div>
      <div className="cd-fallback card">
        <div className="cd-params">
          {["temperature", "top_p", "penalty", "max_tokens"].map((p) => <span key={p}>{p}</span>)}
        </div>
        <hr className="rule" style={{ margin: "16px 0" }} />
        <div className="cd-chain"><strong>Primary</strong><span>→</span><strong>Backup A</strong><span>→</span><strong>Backup B</strong></div>
      </div>
    </div>
  );
}
