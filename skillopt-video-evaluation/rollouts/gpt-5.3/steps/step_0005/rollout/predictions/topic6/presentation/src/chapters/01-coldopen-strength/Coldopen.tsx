import type { ChapterStepProps } from "../../registry/types";
import "./Coldopen.css";

export default function ColdopenChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="t6c-scene scene-pad">
        <div className="t6c-kicker">GPT-IMAGE-2 / 2026 RELEASE</div>
        <div className="t6c-hero-wrap">
          <div>
            <h1 className="t6c-h1 serif-cn">断层领先，不只是参数升级</h1>
            <p className="t6c-sub">Arena.AI：1512 分，领先第二名 242 分</p>
            <div className="t6c-rank">
              <div className="t6c-rank-item"><span>GPT-Image-2</span><b>1512</b></div>
              <div className="t6c-rank-item"><span>第二名</span><b>1270</b></div>
            </div>
          </div>
          <img className="t6c-img" src={`${import.meta.env.BASE_URL}assets/wechat-01.jpg`} alt="wechat source" />
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="t6c-scene scene-pad">
        <div className="t6c-kicker">TEXT RENDER</div>
        <div className="t6c-compare">
          <div className="card t6c-card">
            <div className="label-mono">以前常见</div>
            <p className="t6c-bad">中文多字海报易乱码</p>
            <div className="t6c-noise">X 码 / 错字 / 重影</div>
          </div>
          <div className="t6c-arrow">→</div>
          <div className="card t6c-card t6c-card-accent">
            <div className="label-mono">GPT-Image-2</div>
            <p className="t6c-good">海报、菜单、信息图文本更稳定</p>
            <div className="t6c-lines">
              <span />
              <span />
              <span />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="t6c-scene scene-pad">
        <div className="t6c-kicker">INSTRUCTION FOLLOWING</div>
        <div className="t6c-brief card">
          <div className="t6c-brief-head">Creative Brief</div>
          <ul>
            <li>主体位置：右侧 1/3</li>
            <li>背景：冷色科技感</li>
            <li>文案：标题 + 副标题 + CTA</li>
            <li>禁改项：品牌 Logo / 主色值</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="t6c-scene scene-pad">
      <div className="t6c-kicker">EDITING LOOP</div>
      <div className="t6c-flow">
        <div className="card t6c-node">输入参考图</div>
        <div className="t6c-link" />
        <div className="card t6c-node">局部替换</div>
        <div className="t6c-link" />
        <div className="card t6c-node">风格统一</div>
        <div className="t6c-link" />
        <div className="card t6c-node t6c-node-accent">输出可用图</div>
      </div>
      <img className="t6c-img-mini" src={`${import.meta.env.BASE_URL}assets/wechat-02.png`} alt="wechat sample" />
    </div>
  );
}
