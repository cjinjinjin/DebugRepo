import type { ChapterStepProps } from "../../registry/types";
import "./AccessRoutes.css";

const routes = ["ChatGPT", "Codex", "Lovart", "OpenAI API", "OpenRouter", "302.AI"];

export default function AccessRoutesChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="t6w-scene scene-pad">
        <div className="t6w-kicker">ACCESS MAP</div>
        <h2 className="t6w-title serif-cn">三条入口：官方、平台、API</h2>
        <div className="t6w-grid">
          <div className="card t6w-col"><b>官方</b><p>ChatGPT / Codex</p></div>
          <div className="card t6w-col"><b>平台</b><p>Lovart ChatCanvas</p></div>
          <div className="card t6w-col"><b>API</b><p>OpenAI / OpenRouter / 302.AI</p></div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="t6w-scene scene-pad">
        <div className="t6w-kicker">OFFICIAL ENTRY</div>
        <div className="t6w-dual">
          <div className="card t6w-box"><h3>ChatGPT</h3><p>订阅即用，最快上手</p></div>
          <div className="card t6w-box t6w-box-accent"><h3>Codex</h3><p>写代码同时生成视觉资产</p></div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="t6w-scene scene-pad">
        <div className="t6w-kicker">COLLAB CANVAS</div>
        <div className="t6w-canvas card">
          <div className="t6w-node">需求</div>
          <div className="t6w-node">GPT-Image-2</div>
          <div className="t6w-node">其他模型</div>
          <div className="t6w-node">成品画布</div>
        </div>
        <img className="t6w-img" src={`${import.meta.env.BASE_URL}assets/wechat-03.webp`} alt="lovart-like" />
      </div>
    );
  }

  return (
    <div className="t6w-scene scene-pad">
      <div className="t6w-kicker">API ROUTING</div>
      <div className="t6w-route-wrap">
        {routes.map((name, idx) => (
          <div key={name} className={`t6w-pill ${idx === 0 ? "is-accent" : ""}`}>{name}</div>
        ))}
      </div>
      <div className="t6w-note">统一格式、可切换、按量付费路径更灵活</div>
    </div>
  );
}
