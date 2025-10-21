import torch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests, re, os
from transformers import AutoImageProcessor, AutoModel
import fire
from typing import List, Tuple, Union


# ---------- Load Image ----------
def load_image(image: str, http_url: str = None) -> Union[Image.Image, None]:
    try:
        if http_url is not None:
            url = http_url.format(image)
            resp = requests.get(url, timeout=600)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return image
    except Exception as e:
        print(f"[Warning] Cannot load image {image}: {e}")
        return None


# ---------- Batch Feature Extraction ----------
def process_batch(model, processor, batch_images: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入一批图像，返回 (CLS features, Patch features mean pooled)
    """
    inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state  # [B, 1+num_patches, hidden_dim]
    cls_tokens = last_hidden_states[:, 0, :]  # [B, hidden_dim]

    # 平均所有patch特征（不含CLS与register tokens）
    patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
    # patch_mean = patch_features_flat.mean(dim=1)  # [B, hidden_dim]

    return cls_tokens.cpu().numpy(), patch_features_flat.cpu().numpy()


# ---------- Main Batch Infer ----------
def infer_dinov3_batch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]] = "*",
    image_col: str = "image_path",
    model_path: str = "dinov3-vitb16-pretrain-lvd1689m",
    batch_size: int = 8,
    http_url: str = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    df = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )
    assert image_col in df.columns, f"{image_col} not found in input file"

    # Load model
    print(f"[Load Model] {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="auto")
    model.eval()

    all_cls_strs = []
    all_patch_strs = []

    buffer_images = []
    buffer_indices = []

    for idx, row in df.iterrows():
        img_path = row[image_col]
        image = load_image(img_path, http_url=http_url)

        if image is not None:
            buffer_images.append(image)
            buffer_indices.append(idx)
        else:
            # 如果无法加载图片则填充空特征
            dim = model.config.hidden_size
            all_cls_strs.append(",".join(["0"] * dim))
            all_patch_strs.append(",".join(["0"] * dim))

        if len(buffer_images) == batch_size or idx == len(df) - 1:
            if buffer_images:
                cls_feats, patch_feats = process_batch(model, processor, buffer_images)
                for i in range(len(buffer_images)):
                    cls_str = ",".join(map(lambda x: f"{x:.6f}", cls_feats[i].flatten()))
                    patch_str = ",".join(map(lambda x: f"{x:.6f}", patch_feats[i].flatten()))
                    all_cls_strs.append(cls_str)
                    all_patch_strs.append(patch_str)

                buffer_images.clear()
                buffer_indices.clear()

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

    df["cls_embedding"] = all_cls_strs
    df["patch_embedding"] = all_patch_strs

    df.to_csv(output_file, sep="\t", index=False, quoting=3)
    print(f"[Done] Saved embeddings to: {output_file}")


if __name__ == "__main__":
    fire.Fire(infer_dinov3_batch)

# Example usage:
# python dinov3.py --data_file ~/test.txt --http_url None --output_file ./output.txt --names image --image_col image 
# --model_path /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/DinoV3/dinov3-vitb16-pretrain-lvd1689m/