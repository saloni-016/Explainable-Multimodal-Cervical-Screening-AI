import os
import json
import torch
import numpy as np

from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset
from torch import nn, optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "sipakmed_images")
)

if not os.path.exists(DATA_DIR):
    print("ERROR: sipakmed_images folder was not found.")
    print(DATA_DIR)
    raise SystemExit


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("Loading SIPAKMED dataset...")

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform
)

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)


if len(dataset.classes) != 5:
    print("WARNING: Expected 5 classes, found:", len(dataset.classes))


print("\nImages in each class:")

for class_name, class_index in dataset.class_to_idx.items():
    count = sum(
        1 for label in dataset.targets
        if label == class_index
    )
    print(class_name, ":", count)


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


print("\nDataset split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))


train_dataset = Subset(
    dataset,
    train_indices
)


test_dataset_full = datasets.ImageFolder(
    root=DATA_DIR,
    transform=test_transform
)


val_dataset = Subset(
    test_dataset_full,
    val_indices
)

test_dataset = Subset(
    test_dataset_full,
    test_indices
)


BATCH_SIZE = 32


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", DEVICE)


weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(
    weights=weights
)

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    len(dataset.classes)
)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.00001,
    weight_decay=0.01
)


EPOCHS = 10

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = 100 * correct / total


    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = 100 * val_correct / val_total

    average_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


print("\nTesting model...")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(
    "\nTest Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.classes
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "vit_sipakmed.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


CLASS_PATH = os.path.join(
    BASE_DIR,
    "sipakmed_classes.json"
)

with open(CLASS_PATH, "w") as file:
    json.dump(
        dataset.classes,
        file
    )


print("\nModel saved at:")
print(MODEL_PATH)

print("\nClasses saved at:")
print(CLASS_PATH)

print("\nTraining completed successfully.")