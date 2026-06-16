import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "After go-live, reliability depends on strict SI and Prod separation.",
  "Treat certificates as lifecycle assets with explicit environment matching and expiry planning.",
  "Kusto visibility follows cert context, so cert and namespace mismatches can hide traffic in your target environment.",
  "Fix logging anti-patterns: no silent EventHub exception swallow, use formatted messages, and flush on crash paths.",
  "Recurring deployment incidents include OOM fallback, CUDA UUID mapping, read-only model paths, and ACL mismatches.",
  "Final readiness checklist is image tag, Gen2 integrity, Polaris metrics, ACL, endpoint format, certificate expiry, and Kusto visibility.",
];
