"""
Occlusion-based explanation for the abnormality / risk indicator.

Shows which regions of the cell image drive the result. This is a screening
aid only. It is NOT a cancer diagnosis.

Requires predict_risk.py and sipakmed_risk_config.json in the same folder.
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from train_vit import (
    model,
    transform,
    DEVICE,
    CLASSES,
    ABNORMAL_IDX,
    FLAG_THRESHOLD,
    risk_level,
    RECOMMENDATION,
)
from train_vit import DISCLAIMER

GRID_SIZE = 14      # 14 x 14 patches for a 224 x 224 ViT-B/16 input
PATCH_SIZE = 16
BATCH_SIZE = 32

# 3 x 3 region layout over the 14 x 14 grid
ROW_BOUNDS = [(0, 5), (5, 10), (10, 14)]
COL_BOUNDS = [(0, 5), (5, 10), (10, 14)]
REGION_NAMES = [
    ["Upper-left region", "Upper-central region", "Upper-right region"],
    ["Middle-left region", "Central region", "Middle-right region"],
    ["Lower-left region", "Lower-central region", "Lower-right region"],
]


def get_probabilities(images):
    """Softmax probabilities for a list of PIL images (batched)."""
    outputs = []
    for start in range(0, len(images), BATCH_SIZE):
        chunk = images[start:start + BATCH_SIZE]
        batch = torch.stack([transform(img) for img in chunk]).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        outputs.append(probs)
    return np.concatenate(outputs)


def explain_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))

    base_probs = get_probabilities([image])[0]
    predicted_index = int(base_probs.argmax())
    abnormality_score = float(base_probs[ABNORMAL_IDX].sum())
    level = risk_level(abnormality_score)

    # Explain the abnormality score when the image is flagged;
    # otherwise explain the predicted (normal/benign) cell type.
    if abnormality_score >= FLAG_THRESHOLD:
        target_indices = ABNORMAL_IDX
        explained_target = "abnormality score"
    else:
        target_indices = [predicted_index]
        explained_target = CLASSES[predicted_index].replace("im_", "")

    base_value = float(base_probs[target_indices].sum())

    # Occlude each patch and measure the drop in the target value
    occluded_images = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            occluded = image.copy()
            x0, y0 = col * PATCH_SIZE, row * PATCH_SIZE
            ImageDraw.Draw(occluded).rectangle(
                [x0, y0, x0 + PATCH_SIZE - 1, y0 + PATCH_SIZE - 1],
                fill=(0, 0, 0),
            )
            occluded_images.append(occluded)

    occluded_probs = get_probabilities(occluded_images)
    occluded_values = occluded_probs[:, target_indices].sum(axis=1)

    importance_map = (base_value - occluded_values).reshape(
        GRID_SIZE, GRID_SIZE
    )

    positive_map = np.maximum(importance_map, 0)
    total = positive_map.sum()

    contribution_map = (
        positive_map / total * 100 if total > 0 else np.zeros_like(positive_map)
    )
    normalized_map = (
        positive_map / positive_map.max()
        if positive_map.max() > 0
        else positive_map
    )

    heatmap = Image.fromarray(np.uint8(normalized_map * 255)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(np.array(heatmap), alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.title(
        f"Risk indicator: {level} "
        f"(abnormality score {abnormality_score * 100:.1f}%)"
    )
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Regional contribution summary
    regions = {}
    for i, (r0, r1) in enumerate(ROW_BOUNDS):
        for j, (c0, c1) in enumerate(COL_BOUNDS):
            regions[REGION_NAMES[i][j]] = contribution_map[r0:r1, c0:c1].sum()

    top_regions = [
        {"name": name, "contribution": round(float(value), 2)}
        for name, value in sorted(
            regions.items(), key=lambda x: x[1], reverse=True
        )[:3]
        if value > 0
    ]

    return {
        "predicted_cell_type": CLASSES[predicted_index].replace("im_", ""),
        "abnormality_score": round(abnormality_score * 100, 2),
        "risk_indicator": level,
        "recommendation": RECOMMENDATION[level],
        "explained_target": explained_target,
        "top_regions": top_regions,
        "heatmap_path": output_path,
        "disclaimer": DISCLAIMER,
        "confidence": round(
    float(base_probs[predicted_index]) * 100, 2
),
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python explain_risk.py <image_path> <output_heatmap.png>")
        raise SystemExit

    print("Usage: python explain_vit.py <image_path> <output_heatmap.png>")