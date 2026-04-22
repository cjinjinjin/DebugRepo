"""
Client for PicassoAdsCreative.ZImage-V1-Jinjin DLIS service.

Request format: JSON
  - prompt: str (text description for image generation)
  - width: int (image width, default 1344)
  - height: int (image height, default 768)

Response format:
  {"image": "<base64 PNG>", "width": 1344, "height": 768, "seed": 42}

Auth: AAD Bearer token via MSAL + PFX certificate
"""

import argparse
import base64
import json
import os
import time
import uuid

import msal
import requests
import urllib3
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.hazmat.primitives.hashes import SHA1

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ZImage-V1-Jinjin"

TENANT_ID = os.environ.get("TENANT_ID", "975f013f-7f24-47e8-a7d3-abc4752bf346")
CLIENT_ID = os.environ.get("CLIENT_ID", "4fb0213c-e983-4210-b494-7b73d650e331")
CERT_PATH = os.environ.get("CERT_PATH", "/home/jinjinchen/data/pfx_cert/AggSvcAuthCert-prod.pfx")
DLIS_SCOPE = os.environ.get("DLIS_SCOPE", "e65e832b-d26e-4d59-be94-d261cd10435c/.default")

_msal_app = None


def _load_pfx(pfx_path: str, password=None) -> dict:
    """Load PFX and return MSAL-compatible cert dict with private key + chain."""
    with open(pfx_path, "rb") as f:
        pfx_data = f.read()
    private_key, certificate, additional_certs = load_key_and_certificates(
        pfx_data, password, default_backend()
    )
    key_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    cert_pem = certificate.public_bytes(Encoding.PEM)
    chain = [cert_pem]
    if additional_certs:
        chain += [c.public_bytes(Encoding.PEM) for c in additional_certs]
    return {
        "private_key": key_pem,
        "thumbprint": certificate.fingerprint(SHA1()).hex().upper(),
        "public_certificate": b"".join(chain).decode(),
    }


def _get_bearer_token() -> str:
    global _msal_app
    if _msal_app is None:
        cert = _load_pfx(CERT_PATH)
        authority = f"https://login.microsoftonline.com/{TENANT_ID}"
        _msal_app = msal.ConfidentialClientApplication(
            client_id=CLIENT_ID,
            authority=authority,
            client_credential={
                "private_key": cert["private_key"].decode(),
                "thumbprint": cert["thumbprint"],
                "public_certificate": cert["public_certificate"],
            },
        )
    result = _msal_app.acquire_token_for_client(scopes=[DLIS_SCOPE])
    if "access_token" not in result:
        raise RuntimeError(f"Failed to get token: {result.get('error_description', result)}")
    return f"Bearer {result['access_token']}"


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

    bearer_token = _get_bearer_token()
    headers = {
        "Authorization": bearer_token,
        "Content-Type": "application/json",
    }

    st = time.perf_counter()
    resp = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        verify=False,
        timeout=timeout_s,
    )
    latency_ms = (time.perf_counter() - st) * 1000

    print(f"Status: {resp.status_code}  Latency: {latency_ms:.0f}ms")

    if resp.status_code == 200:
        result = resp.json()
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
    parser.add_argument("--cert", default=None, help="Override PFX cert path")
    args = parser.parse_args()

    if args.cert:
        global CERT_PATH
        CERT_PATH = args.cert

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
