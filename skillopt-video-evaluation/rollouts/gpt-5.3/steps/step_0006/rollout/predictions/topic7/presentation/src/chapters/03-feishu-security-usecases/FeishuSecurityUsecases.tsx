import type { ChapterStepProps } from "../../registry/types";
import "./FeishuSecurityUsecases.css";

const FLOW = [
  "建应用",
  "开权限",
  "发布",
  "配通道",
  "事件订阅",
  "联调",
];

const CASES = ["晨间简报", "邮件分流", "手机 DevOps", "知识库问答"];

export default function FeishuSecurityUsecases({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="fsu-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">03 / Feishu</span></div>
        <div className="fsu-flow">
          {FLOW.map((item, i) => (
            <div key={item} className="fsu-node card" style={{ animationDelay: `${i * 70}ms` }}>
              <span className="label-mono">STEP {i + 1}</span>
              <strong>{item}</strong>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="fsu-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">03 / Skills</span></div>
        <div className="fsu-dual">
          <section className="fsu-panel card">
            <p className="kicker">MCP / Capability</p>
            <h3>工具能力层</h3>
            <ul>
              <li>文件、浏览器、代码、消息通道</li>
              <li>定义可调用边界与权限</li>
              <li>失败时返回可解释错误</li>
            </ul>
          </section>
          <section className="fsu-panel fsu-panel-accent card">
            <p className="kicker">Skills / Orchestration</p>
            <h3>流程编排层</h3>
            <ul>
              <li>把高频任务拆成可复用 SOP</li>
              <li>定义调用顺序、重试与回退</li>
              <li>保证跨步骤执行一致性</li>
            </ul>
          </section>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="fsu-scene scene-pad">
        <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">03 / Security</span></div>
        <div className="fsu-security card">
          <div className="fsu-rule"><span>01</span><b>最小权限默认拒绝</b><em>仅授予任务必需能力</em></div>
          <div className="fsu-rule"><span>02</span><b>高风险动作显式确认</b><em>支付 / 删除 / 安装 / 外发</em></div>
          <div className="fsu-rule"><span>03</span><b>供应链与插件审计</b><em>混淆或不可审计内容先停</em></div>
          <div className="fsu-rule"><span>04</span><b>凭证治理</b><em>密钥不进日志、不进提交</em></div>
        </div>
      </div>
    );
  }

  return (
    <div className="fsu-scene scene-pad">
      <div className="masthead"><span className="brand">OpenClaw Focus</span><span className="issue">03 / Value</span></div>
      <div className="fsu-grid">
        {CASES.map((name) => (
          <article key={name} className="fsu-case card">
            <div className="label-mono">Use Case</div>
            <h4>{name}</h4>
            <p>目标：把一次性问答升级成稳定任务链执行。</p>
          </article>
        ))}
      </div>
      <div className="fsu-close pull-quote">先跑通基础链路，再逐步放权。</div>
    </div>
  );
}
