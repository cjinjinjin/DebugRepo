"""
Client for PicassoAdsCreative.Gemma4-AWQ-v2 DLIS endpoint.

Usage:
  python test_gemma4_dlis.py
  python test_gemma4_dlis.py --cert /path/to/cert.pfx
  python test_gemma4_dlis.py --query-file test_query.txt
"""

import argparse
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

API_URL = "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.Gemma4-AWQ-v2"

TENANT_ID = os.environ.get("TENANT_ID", "975f013f-7f24-47e8-a7d3-abc4752bf346")
CLIENT_ID = os.environ.get("CLIENT_ID", "4fb0213c-e983-4210-b494-7b73d650e331")
CERT_PATH = os.environ.get("CERT_PATH", r"C:\Users\jinjinchen\Downloads\AggSvcAuthCert-prod.pfx")
DLIS_SCOPE = os.environ.get("DLIS_SCOPE", "e65e832b-d26e-4d59-be94-d261cd10435c/.default")

_msal_app = None


def _load_pfx(pfx_path: str, password=None) -> dict:
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


def call_gemma4(landing_page_content: str, url: str = "", num_prompts: int = 5, timeout_s: int = 120) -> dict:
    payload = {
        "landing_page_content": landing_page_content,
        "url": url,
        "num_prompts": num_prompts,
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
        status = result.get("Status", "?")
        compliant = result.get("format_compliant", "?")
        prompts = result.get("generated_prompts", [])
        scenes = result.get("scenes", [])
        print(f"Status: {status}  format_compliant: {compliant}")
        print(f"Scenes ({len(scenes)}):")
        for i, s in enumerate(scenes, 1):
            print(f"  {i}. {s}")
        print(f"Prompts ({len(prompts)}):")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p[:120]}{'...' if len(p) > 120 else ''}")
        return result
    else:
        print(f"FAILED: {resp.text[:500]}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Test Gemma4 T2I DLIS endpoint")
    parser.add_argument("--cert", default=None, help="Override PFX cert path")
    parser.add_argument("--query-file", help="Read queries from file (one JSON per line)")
    parser.add_argument("--url", default="https://trailmaster.example.com")
    parser.add_argument("--num-prompts", type=int, default=5)
    args = parser.parse_args()

    if args.cert:
        global CERT_PATH
        CERT_PATH = args.cert

    if args.query_file:
        with open(args.query_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            payload = json.loads(line)
            lp = payload.get("landing_page_content", "")
            url = payload.get("url", "")
            n = payload.get("num_prompts", 5)
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(lines)}] url={url[:60]}  lp={lp[:60]}...")
            call_gemma4(lp, url, n)
    else:
        lp = ("Welcome to TrailMaster Outdoor Gear. Premium hiking boots, "
              "ultralight backpacks, and camping essentials for your next "
              "adventure. Free shipping on orders over $99.")
        print(f"Testing with default query: {lp[:60]}...")
        call_gemma4(lp, args.url, args.num_prompts)


if __name__ == "__main__":
    main()
