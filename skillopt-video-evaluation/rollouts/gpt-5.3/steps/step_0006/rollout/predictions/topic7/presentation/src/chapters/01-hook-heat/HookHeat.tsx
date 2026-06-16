import type { ChapterStepProps } from "../../registry/types";
import "./HookHeat.css";

const METRICS = [
  { key: "Stars", value: "297K" },
  { key: "Community", value: "116K+" },
  { key: "Contrib", value: "1000+" },
];

export default function HookHeat({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="hh-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">01 / Hook</span></div>
        <div className="hh-grid" />
        <div className="hh-hero card">
          <div className="kicker">Heat Signal</div>
          <div className="hh-number hero-num">1亿+</div>
          <p>这不是普通热点，是开发者工作流级别的迁移信号。</p>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="hh-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">01 / Hook</span></div>
        <div className="hh-metrics">
          {METRICS.map((m) => (
            <div key={m.key} className="hh-card card">
              <div className="label-mono">{m.key}</div>
              <div className="hh-value hero-num">{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="hh-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">01 / Hook</span></div>
        <div className="hh-compare">
          <section className="hh-pane card">
            <h3>聊天工具</h3>
            <ul><li>回合式问答</li><li>难以持续执行</li><li>上下文短</li></ul>
          </section>
          <div className="hh-vs">VS</div>
          <section className="hh-pane hh-pane-accent card">
            <h3>OpenClaw</h3>
            <ul><li>长期记忆</li><li>工具调度</li><li>流程复用</li></ul>
          </section>
        </div>
      </div>
    );
  }

  return (
    <div className="hh-scene scene-pad">
      <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">01 / Hook</span></div>
      <div className="hh-flow card">
        <div className="hh-node">消息入口</div>
        <div className="hh-arrow">→</div>
        <div className="hh-node hh-core">OpenClaw</div>
        <div className="hh-arrow">→</div>
        <div className="hh-node">工具执行层</div>
      </div>
    </div>
  );
}
