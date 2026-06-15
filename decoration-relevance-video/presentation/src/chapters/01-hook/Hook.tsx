import type { ChapterStepProps } from "../../registry/types";
import "./Hook.css";

/**
 * Chapter 01 — Hook: What Is Decoration Relevance?
 *
 * 5 steps:
 *   0  Bare ad card + hook question
 *   1  Decorated ad with labeled decoration types
 *   2  Match vs mismatch comparison
 *   3  Four sub-tasks (Task 0/1/2/3)
 *   4  Unified model architecture diagram
 */
export default function Hook({ step }: ChapterStepProps) {
  /* ── step 0: bare ad with hook question ── */
  if (step === 0) {
    return (
      <div className="hk-wrap">
        <div className="hk-ad-card card">
          <div className="hk-ad-url">ad &middot; www.contoso-cloud.com</div>
          <div className="hk-ad-headline">
            Contoso Cloud &mdash; Enterprise Solutions
          </div>
          <div className="hk-ad-desc">
            Scalable cloud infrastructure for modern businesses. Get started
            with a free trial today.
          </div>
        </div>
        <div className="hk-hook-q">
          Why do some ads have extra links?
        </div>
      </div>
    );
  }

  /* ── step 1: decorated ad with labeled types ── */
  if (step === 1) {
    return (
      <div className="hk-wrap">
        <div className="hk-ad-card card">
          <div className="hk-ad-url">ad &middot; www.contoso-cloud.com</div>
          <div className="hk-ad-headline">
            Contoso Cloud &mdash; Enterprise Solutions
          </div>
          <div className="hk-ad-desc">
            Scalable cloud infrastructure for modern businesses. Get started
            with a free trial today.
          </div>

          <div className="hk-deco-section">
            <div className="hk-deco-row">
              <span className="hk-deco-tag">
                <span className="hk-deco-dot" /> Sitelinks
              </span>
              <span className="hk-deco-tag">
                <span className="hk-deco-dot" /> Callouts
              </span>
              <span className="hk-deco-tag">
                <span className="hk-deco-dot" /> Snippets
              </span>
              <span className="hk-deco-tag">
                <span className="hk-deco-dot" /> Price Ext.
              </span>
            </div>
            <div className="hk-sitelinks">
              <div className="hk-sitelink-item">Pricing Plans</div>
              <div className="hk-sitelink-item">Free Trial</div>
              <div className="hk-sitelink-item">Enterprise FAQ</div>
              <div className="hk-sitelink-item">Contact Sales</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── step 2: match vs mismatch ── */
  if (step === 2) {
    return (
      <div className="hk-wrap">
        <div style={{ width: "100%", maxWidth: 1500 }}>
          <div className="hk-match-hero">
            <div className="hk-match-title">
              Does this decoration match the query?
            </div>
            <div className="hk-match-sub">
              Decoration Relevance &mdash; the core question
            </div>
          </div>

          <div className="hk-match-layout">
            {/* good match */}
            <div className="hk-match-col hk-match-good card">
              <div className="hk-match-label hk-match-label-good">
                Matching
              </div>
              <div className="hk-match-query">
                Query: <span>&ldquo;cloud hosting pricing&rdquo;</span>
              </div>
              <div className="hk-match-deco">
                Sitelink: &ldquo;Pricing Plans &mdash; Compare Tiers&rdquo;
              </div>
              <div className="hk-match-verdict hk-verdict-pass">
                Verdict: NonBad &rarr; keep
              </div>
            </div>

            {/* bad match */}
            <div className="hk-match-col hk-match-bad card">
              <div className="hk-match-label hk-match-label-bad">
                Mismatching
              </div>
              <div className="hk-match-query">
                Query: <span>&ldquo;cloud hosting pricing&rdquo;</span>
              </div>
              <div className="hk-match-deco">
                Sitelink: &ldquo;Careers at Contoso &mdash; Join Our
                Team&rdquo;
              </div>
              <div className="hk-match-verdict hk-verdict-fail">
                Verdict: Bad &rarr; filter out
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── step 3: four sub-tasks ── */
  if (step === 3) {
    const tasks = [
      {
        id: "0",
        name: "Sitelinks",
        items: "SiteLink\nSLAB",
        deprecated: false,
      },
      {
        id: "1",
        name: "Long Desc.",
        items: "LongDescription\nDynamicDescription",
        deprecated: true,
      },
      {
        id: "2",
        name: "Snippets",
        items: "StructuredSnippetExt.\nSmartCategory",
        deprecated: false,
      },
      {
        id: "3",
        name: "Callouts",
        items: "CalloutExtension\nPriceExtension",
        deprecated: false,
      },
    ];

    return (
      <div className="hk-wrap">
        <div className="hk-tasks-layout">
          <div className="hk-tasks-title">Four Sub-Tasks</div>
          <div className="hk-tasks-grid">
            {tasks.map((t) => (
              <div
                key={t.id}
                className={`hk-task-box card${t.deprecated ? " hk-deprecated" : ""}`}
              >
                <div className="hk-task-id hero-num">T{t.id}</div>
                <div className="hk-task-name">{t.name}</div>
                <div className="hk-task-items">
                  {t.items.split("\n").map((line, i) => (
                    <div key={i}>{line}</div>
                  ))}
                </div>
                {t.deprecated && (
                  <div className="hk-deprecated-tag">Deprecated</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ── step 4: unified model diagram ── */
  if (step === 4) {
    const heads = [
      { id: "T0", name: "Sitelinks", dim: false },
      { id: "T1", name: "Long Desc.", dim: true },
      { id: "T2", name: "Snippets", dim: false },
      { id: "T3", name: "Callouts", dim: false },
    ];

    return (
      <div className="hk-wrap">
        <div className="hk-arch-layout">
          <div className="hk-arch-title">One Model, Four Heads</div>

          <div className="hk-arch-diagram">
            {/* shared backbone */}
            <div className="hk-arch-core card">
              <div className="hk-arch-core-label">Shared Backbone</div>
              <div className="hk-arch-core-name">TriBert V9</div>
              <div className="hk-arch-core-sub">
                BERT encoder &middot; multi-task
              </div>
            </div>

            {/* connectors */}
            <div className="hk-arch-connectors">
              <div className="hk-arch-horizontal" />
              <div className="hk-arch-line" />
              <div className="hk-arch-line" />
              <div className="hk-arch-line" />
              <div className="hk-arch-line" />
            </div>

            {/* classification heads */}
            <div className="hk-arch-heads">
              {heads.map((h) => (
                <div
                  key={h.id}
                  className={`hk-arch-head${!h.dim ? " hk-head-active" : " hk-head-dim"}`}
                >
                  <div className="hk-arch-head-id">{h.id}</div>
                  <div className="hk-arch-head-name">{h.name}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="hk-arch-bottom-label">
            Same encoder &middot; per-task classification
          </div>
        </div>
      </div>
    );
  }

  return null;
}
