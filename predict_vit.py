import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

MODEL_PATH = "vit_sipakmed.pth"
CLASSES_PATH = "sipakmed_classes.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

model = vit_b_16()
model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(classes))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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

image_path = input("Enter the path of the cervical-cell image: ").strip().strip('"')

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

predicted_class = classes[predicted.item()].replace("im_", "")
confidence_percentage = confidence.item() * 100

print("\n" + "=" * 50)
print("CERVICAL CELL IMAGE RESULT")
print("=" * 50)

print(f"\nPredicted cell class: {predicted_class}")
print(f"Confidence: {confidence_percentage:.2f}%")