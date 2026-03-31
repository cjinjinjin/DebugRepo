"""
Generate images using gpt-image-1 via Papyrus API.

Authentication options:
  1. Azure AD token (recommended for internal use): set USE_AZURE_AD=true
  2. API key: set PAPYRUS_API_KEY environment variable

Usage:
  python papyrus_generate_image.py --prompt "a cat sitting on a cloud" --output cat.png
"""

import argparse
import base64
import os
import requests


PAPYRUS_ENDPOINT = "https://WestUS2Eval.papyrus.binginternal.com"
MODEL_NAME = "gpt-image-1-2025-04-15-Eval"


def get_azure_ad_token() -> str:
    """Get Azure AD token using azure-identity library."""
    try:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        token = credential.get_token("https://api.bing.microsoft.com/.default")
        return token.token
    except ImportError:
        raise RuntimeError(
            "azure-identity not installed. Run: pip install azure-identity\n"
            "Or set PAPYRUS_API_KEY instead."
        )


def get_auth_headers() -> dict:
    use_azure_ad = os.environ.get("USE_AZURE_AD", "").lower() in ("1", "true", "yes")
    api_key = os.environ.get("PAPYRUS_API_KEY", "")

    if use_azure_ad:
        token = get_azure_ad_token()
        return {"Authorization": f"Bearer {token}"}
    elif api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        # Try Azure AD as default
        try:
            token = get_azure_ad_token()
            return {"Authorization": f"Bearer {token}"}
        except Exception:
            raise RuntimeError(
                "No authentication configured.\n"
                "Options:\n"
                "  1. Set USE_AZURE_AD=true (requires azure-identity: pip install azure-identity)\n"
                "  2. Set PAPYRUS_API_KEY=<your_key>"
            )


def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1) -> list[str]:
    """
    Call Papyrus image generation API and return list of base64-encoded images.
    """
    url = f"{PAPYRUS_ENDPOINT}/v1/images/generations"

    headers = {
        "Content-Type": "application/json",
        "papyrus-model-name": MODEL_NAME,
        **get_auth_headers(),
    }

    payload = {
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": n,
        "response_format": "b64_json",
    }

    print(f"Calling {url}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt!r}")

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if not response.ok:
        raise RuntimeError(
            f"API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return [item["b64_json"] for item in data["data"]]


def save_images(b64_images: list[str], output_prefix: str) -> list[str]:
    saved = []
    for i, b64 in enumerate(b64_images):
        if len(b64_images) == 1:
            path = output_prefix if output_prefix.endswith(".png") else f"{output_prefix}.png"
        else:
            base = output_prefix.removesuffix(".png")
            path = f"{base}_{i + 1}.png"

        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate images via Papyrus gpt-image-1")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path (default: output.png)")
    parser.add_argument("--size", default="1024x1024", choices=["1024x1024", "1792x1024", "1024x1792"],
                        help="Image size")
    parser.add_argument("--quality", default="standard", choices=["standard", "hd"],
                        help="Image quality")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    args = parser.parse_args()

    b64_images = generate_image(
        prompt=args.prompt,
        size=args.size,
        quality=args.quality,
        n=args.n,
    )
    save_images(b64_images, args.output)


if __name__ == "__main__":
    main()
