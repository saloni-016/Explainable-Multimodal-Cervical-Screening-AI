
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

MODEL_PATH = "vit_sipakmed.pth"
CLASSES_PATH = "sipakmed_classes.json"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

model = vit_b_16()

model.heads.head = torch.nn.Linear(
    model.heads.head.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def explain_image(image_path, output_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize((224, 224))

    original_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        original_output = model(
            original_tensor
        )

        original_probabilities = torch.softmax(
            original_output,
            dim=1
        )

        predicted_index = (
            original_probabilities
            .argmax(dim=1)
            .item()
        )

        original_confidence = (
            original_probabilities[
                0,
                predicted_index
            ].item()
        )

    grid_size = 14
    patch_size = 16

    importance_map = np.zeros(
        (grid_size, grid_size)
    )

    for row in range(grid_size):

        for col in range(grid_size):

            occluded_image = image.copy()

            pixels = occluded_image.load()

            start_x = col * patch_size
            start_y = row * patch_size

            for y in range(
                start_y,
                start_y + patch_size
            ):

                for x in range(
                    start_x,
                    start_x + patch_size
                ):

                    pixels[x, y] = (
                        0,
                        0,
                        0
                    )

            occluded_tensor = transform(
                occluded_image
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(
                    occluded_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                new_confidence = (
                    probabilities[
                        0,
                        predicted_index
                    ].item()
                )

            importance_map[
                row,
                col
            ] = (
                original_confidence
                - new_confidence
            )

    positive_map = np.maximum(
        importance_map,
        0
    )

    total_contribution = positive_map.sum()

    if total_contribution > 0:

        contribution_map = (
            positive_map
            / total_contribution
        ) * 100

    else:

        contribution_map = np.zeros_like(
            positive_map
        )

    if positive_map.max() > 0:

        normalized_map = (
            positive_map
            / positive_map.max()
        )

    else:

        normalized_map = positive_map

    heatmap = Image.fromarray(
        np.uint8(
            normalized_map * 255
        )
    )

    heatmap = heatmap.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(image)

    plt.imshow(
        heatmap_array,
        alpha=0.5,
        cmap="jet"
    )

    plt.axis("off")

    class_name = classes[
        predicted_index
    ].replace("im_", "")

    confidence_percentage = (
        original_confidence * 100
    )

    plt.title(
        f"ViT Explanation: "
        f"{class_name} "
        f"({confidence_percentage:.2f}%)"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    regions = {

        "Upper-left region":
            contribution_map[
                0:5,
                0:5
            ].sum(),

        "Upper-central region":
            contribution_map[
                0:5,
                5:10
            ].sum(),

        "Upper-right region":
            contribution_map[
                0:5,
                10:14
            ].sum(),

        "Central region":
            contribution_map[
                5:10,
                5:10
            ].sum(),

        "Lower-left region":
            contribution_map[
                10:14,
                0:5
            ].sum(),

        "Lower-central region":
            contribution_map[
                10:14,
                5:10
            ].sum(),

        "Lower-right region":
            contribution_map[
                10:14,
                10:14
            ].sum()
    }

    sorted_regions = sorted(
        regions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_regions = [
        {
            "name": name,
            "contribution": round(
                value,
                2
            )
        }
        for name, value
        in sorted_regions[:3]
        if value > 0
    ]

    return {
        "class_name": class_name,
        "confidence": round(
            confidence_percentage,
            2
        ),
        "top_regions": top_regions,
        "heatmap_path": output_path
    }

