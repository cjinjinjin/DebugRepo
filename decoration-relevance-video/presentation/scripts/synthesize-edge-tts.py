"""
Synthesize audio segments using edge-tts (free Microsoft TTS, no API key).

Usage:
  python scripts/synthesize-edge-tts.py [--voice VOICE] [--force]

Reads audio-segments.json, outputs mp3 to public/audio/<chapter>/<step>.mp3.
Skips existing files unless --force is passed.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("ERROR: edge-tts not installed. Run: python -m pip install edge-tts")
    sys.exit(1)


async def synthesize_one(text: str, out_path: Path, voice: str) -> bool:
    """Synthesize a single segment. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize audio with edge-tts")
    parser.add_argument(
        "--voice",
        type=str,
        default="en-US-GuyNeural",
        help="Edge TTS voice name (default: en-US-GuyNeural). "
             "Other good options: en-US-JennyNeural, en-US-AriaNeural, en-US-DavisNeural",
    )
    parser.add_argument("--force", action="store_true", help="Re-synthesize existing files")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    segments_file = project_dir / "audio-segments.json"
    audio_dir = project_dir / "public" / "audio"

    if not segments_file.exists():
        print("ERROR: audio-segments.json not found. Run: npm run extract-narrations")
        sys.exit(1)

    with open(segments_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    total = len(segments)
    done = 0
    skipped = 0
    failed = 0

    print(f"Synthesizing {total} segments with voice '{args.voice}'")
    print(f"Output: {audio_dir}\n")

    for i, seg in enumerate(segments, 1):
        out_path = audio_dir / seg["audio"]
        label = seg["audio"]

        # Skip empty narrations
        if not seg["text"].strip():
            print(f"[{i:3d}/{total}] {label}   skip (empty)")
            skipped += 1
            continue

        # Skip existing unless --force
        if out_path.exists() and not args.force:
            print(f"[{i:3d}/{total}] {label}   skip (exists)")
            skipped += 1
            continue

        ok = await synthesize_one(seg["text"], out_path, args.voice)
        if ok:
            # Get file size for feedback
            size_kb = out_path.stat().st_size / 1024
            print(f"[{i:3d}/{total}] {label}   OK  {size_kb:.0f}KB")
            done += 1
        else:
            failed += 1

    print(f"\nDone: {done} synthesized, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    asyncio.run(main())
