import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Step one starts in your personal OaaS template branch so you can iterate safely.",
  "Most edits happen in model dot py and dlis_inter dot py, with server changes only when custom payload format is needed.",
  "For build strategy, choose fast iteration Dockerfile or full pinned stack depending on control needs.",
  "The local loop is build image, run container with mapping, send request, and check logs.",
  "On data lane, upload to Gen1 with flat layout and migrate to Gen2 because serving reads from Gen2.",
  "On image lane, push branch, capture CI image tag, and merge only when both lanes are green.",
];
