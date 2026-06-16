import type { ChapterStepProps } from "../../registry/types";
import "./CasePlaybook.css";

const groups = [
  ["UI 样机", "品牌海报", "信息图"],
  ["学术配图", "漫画角色", "技术架构"],
  ["头像贴纸", "地图路线", "产品视觉"],
];

export default function CasePlaybookChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="t6p-scene scene-pad">
        <div className="t6p-kicker">CASE LIBRARY</div>
        <h2 className="t6p-title serif-cn">不是看图站，而是可复现案例库</h2>
        <div className="t6p-metrics">
          <div className="card t6p-m">完整 prompt</div>
          <div className="card t6p-m">对应模板</div>
          <div className="card t6p-m">可改字段</div>
          <div className="card t6p-m">一句话复现</div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="t6p-scene scene-pad">
        <div className="t6p-kicker">CATEGORY BLOCK A</div>
        <div className="t6p-tags">{groups[0].map((g) => <span key={g}>{g}</span>)}</div>
        <img className="t6p-img" src={`${import.meta.env.BASE_URL}assets/wechat-04.webp`} alt="sample" />
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="t6p-scene scene-pad">
        <div className="t6p-kicker">CATEGORY BLOCK B</div>
        <div className="t6p-tags">{groups[1].map((g) => <span key={g}>{g}</span>)}</div>
        <div className="t6p-board card">
          <div>pipeline figure</div><div>graphical abstract</div><div>角色关系图</div>
        </div>
      </div>
    );
  }

  return (
    <div className="t6p-scene scene-pad">
      <div className="t6p-kicker">BOUNDARY</div>
      <div className="t6p-limit card">
        <div className="t6p-big hero-num">PNG</div>
        <div className="t6p-vs">≠</div>
        <div className="t6p-right">可编辑 SVG 源文件</div>
      </div>
      <div className="t6p-note">适合文档配图、技术分享、快速表达；不替代 draw.io / Excalidraw。</div>
    </div>
  );
}
