# DLIS Model Deployment Guide v5.5 — Video Voiceover Script

You can have a strong model and still fail deployment if your pipeline is weak.

---

This guide compresses the real DLIS flow into one practical path: local validation, parallel prep, Polaris gate, production cutover, and operations hardening.

---

Version 5.5 matters because it combines Gemma4 and ZImage rollout lessons with hands-on team debugging notes.

---

Keep one backbone in mind: local test first, then data and image prep in parallel, then Polaris, then production, then verification.

---

If local validation is skipped, every later step becomes slower and more expensive to fix.

---

Step one starts in your personal OaaS_LLMTemplate branch.

---

Most changes happen in model.py and dlis_inter.py, with http_server.py only when custom request formats are needed.

---

For container strategy, you choose either Dockerfile_vllm_fast for speed or a full pinned stack for tighter dependency control.

---

The local run loop is simple: build image, start container with port mapping, send test request.

---

At this stage, validate output correctness and observability together, including Kusto logging, not as separate activities.

---

After local success, split work into two lanes to save time.

---

Data lane: upload checkpoint files to Gen1 with the recommended flat layout.

---

Root-level placement is critical: dlis_inter.py and certificates must be at the top level, or runtime discovery fails.

---

Then migrate from Gen1 to Gen2, because DLIS serving reads model data from Gen2.

---

Image lane runs in parallel: push branch, trigger CI, and capture the emitted image tag.

---

When both lanes are ready, merge at Polaris testing.

---

In Polaris, the high-risk fields are ModelPath, ModelDataPath, environment variables, and model ready timeout.

---

Use loading status as a real gate: one hundred percent plus success.

---

Then verify four dimensions: output quality, latency, stability, and resource usage.

---

If any dimension fails, go back, patch code or config, rebuild, and retest.

---

After Polaris passes, move to production deployment in One Inference Portal.

---

Choose hardware by workload profile: heavier relevance workloads usually need A100-class memory, while smaller diversity workloads can use lighter machines.

---

Configure pages in sequence: Key, Hardware, General, and ACL.

---

Treat ACL as a release blocker, because wrong ACL configuration often causes immediate 403 errors.

---

After submission, verify endpoint naming and URL format, and always validate requests with client certificates.

---

Now comes operations hardening.

---

Keep SI and Prod environments separated; sharing endpoints is explicitly a pilot blocker.

---

Manage certificate lifecycle proactively, including environment matching and expiration.

---

Kusto visibility depends on cert-namespace consistency; mismatches can make valid traffic appear invisible in your target environment.

---

Avoid silent failures in logging: never swallow EventHub exceptions, use record.getMessage, and flush logs on crash paths.

---

Final readiness checklist: image tag, Gen2 integrity, Polaris metrics, ACL correctness, endpoint format, certificate expiry window, and Kusto visibility.

---

If that checklist is green, you are not just deployed — you are operable.
