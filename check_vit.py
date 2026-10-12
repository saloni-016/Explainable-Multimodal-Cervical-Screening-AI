import json,os
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "sipakmed_images")
)
MODEL_PATH = "vit_sipakmed.pth"
CLASSES_PATH = "sipakmed_classes.json"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

targets = np.array(dataset.targets)
indices = np.arange(len(dataset))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=targets
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=targets[temp_indices]
)

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

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

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels.numpy()
        )

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")

print(
    " " * 24 +
    " ".join(
        f"{c.replace('im_', '')[:8]:>10}"
        for c in classes
    )
)

for i, row in enumerate(matrix):

    print(
        f"{classes[i].replace('im_', '')[:20]:>20} "
        + " ".join(
            f"{value:>10}"
            for value in row
        )
    )

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            c.replace("im_", "")
            for c in classes
        ]
    )
)