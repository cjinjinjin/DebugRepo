"""
Client for PicassoAdsCreative.ZImage-V1-Jinjin DLIS service.

Request format: JSON
  - prompt: str (text description for image generation)
  - width: int (image width, default 1344)
  - height: int (image height, default 768)

Response format:
  {"image": "<base64 PNG>", "width": 1344, "height": 768, "seed": 42}

Auth: Client certificate (private1.cer + private1.key)
"""

import argparse
import base64
import json
import os
import time
import uuid

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ZImage-V1-Jinjin"

CERT_DIR = os.environ.get("CERT_DIR", "/home/jinjinchen/dlis/abo-models/team/dai/auto_image/client")
CERT_FILE = os.path.join(CERT_DIR, "private1.cer")
KEY_FILE = os.path.join(CERT_DIR, "private1.key")


def call_zimage(prompt: str, width: int = 1344, height: int = 768, timeout_s: int = 120) -> dict:
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "tracking_data": {
            "requestid": f"jinjinchen-test-{uuid.uuid4().hex[:8]}",
            "trackingid": f"jinjinchen-test-{uuid.uuid4().hex[:8]}",
            "sessionid": "",
            "customerid": "",
            "callername": "jinjinchen_test",
        },
    }

    st = time.perf_counter()
    resp = requests.post(
        API_URL,
        json=payload,
        cert=(CERT_FILE, KEY_FILE),
        headers={"Content-Type": "application/json"},
        verify=False,
        timeout=timeout_s,
    )
    latency_ms = (time.perf_counter() - st) * 1000

    print(f"Status: {resp.status_code}  Latency: {latency_ms:.0f}ms")

    if resp.status_code == 200:
        result = resp.json()
        # Save image if present
        if "image" in result:
            img_bytes = base64.b64decode(result["image"])
            out_path = f"zimage_output_{int(time.time())}.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"Image saved: {out_path} ({len(img_bytes)} bytes)")
            print(f"Resolution: {result.get('width', '?')}x{result.get('height', '?')}, seed={result.get('seed', '?')}")
        else:
            print(f"Response: {resp.text[:500]}")
        return result
    else:
        print(f"FAILED: {resp.text[:500]}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Test ZImage DLIS endpoint")
    parser.add_argument("--prompt", default="A beautiful sunset over the ocean with golden light reflecting on calm waves")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--query-file", help="Read prompts from file (one JSON per line)")
    args = parser.parse_args()

    if args.query_file:
        with open(args.query_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            payload = json.loads(line)
            prompt = payload.get("prompt", "")
            width = payload.get("width", 1344)
            height = payload.get("height", 768)
            print(f"\n[{i+1}/{len(lines)}] prompt={prompt[:60]}...")
            call_zimage(prompt, width, height)
    else:
        call_zimage(args.prompt, args.width, args.height)


if __name__ == "__main__":
    main()
