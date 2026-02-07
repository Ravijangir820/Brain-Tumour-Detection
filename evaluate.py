# =========================================================
# 0. SYSTEM SETUP (WINDOWS SAFE)
# =========================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 1. IMPORTS
# =========================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# =========================================================
# 2. CONFIG
# =========================================================
DATASET_DIR = "./Brain Cancer"
MODEL_DIR = "./trained_models"
TB_DIR = "./tensorboard_eval"
MISCLASS_DIR = "./misclassified"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(MISCLASS_DIR, exist_ok=True)

# =========================================================
# 3. TRANSFORMS
# =========================================================
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# 4. DATASET
# =========================================================
dataset = datasets.ImageFolder(DATASET_DIR, transform=val_tfms)
class_names = dataset.classes
num_classes = len(class_names)

val_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================================================
# 5. MODELS (MATCH TRAINING)
# =========================================================
class VAF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n)
        )
    def forward(self, x): return self.net(x)


class UNET(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, n)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))


class CustomCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def build_resnet50(n):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n)
    return model

# =========================================================
# 6. GRAD-CAM (CORRECT)
# =========================================================
def get_target_layer(model, name):
    if name == "VAF":
        return model.net[2]
    if name == "UNET":
        return model.features[3]
    if name == "CustomCNN":
        return model.features[6]
    if "ResNet50" in name:
        return model.layer4
    raise ValueError("Unknown model")


def generate_gradcam(model, x, target_layer):
    model.eval()
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    out = model(x)
    cls = out.argmax(dim=1)

    model.zero_grad()
    out[0, cls].backward()

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    # ✅ THIS LINE FIXES YOUR ERROR
    return cam.detach().cpu().numpy()[0]


def save_gradcam(model, name, idx):
    img_path, _ = dataset.samples[idx]
    img = Image.open(img_path).convert("RGB")

    x = val_tfms(img).unsqueeze(0).to(DEVICE)
    cam = generate_gradcam(model, x, get_target_layer(model, name))

    cam = Image.fromarray(np.uint8(255 * cam)).resize(img.size)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img); plt.axis("off"); plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.axis("off"); plt.title("Grad-CAM")

    plt.suptitle(f"{name} — Sample {idx}")
    plt.show()

# =========================================================
# 7. EVALUATION
# =========================================================
def evaluate_model(model, name, writer):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(DEVICE)
            out = model(x)
            pr = out.argmax(1).cpu().numpy()

            preds.extend(pr)
            targets.extend(y.numpy())

            for j, (pp, tt) in enumerate(zip(pr, y)):
                if pp != tt:
                    img_path, _ = dataset.samples[i * BATCH_SIZE + j]
                    out_dir = f"{MISCLASS_DIR}/{name}"
                    os.makedirs(out_dir, exist_ok=True)
                    Image.open(img_path).save(
                        f"{out_dir}/{class_names[tt]}_AS_{class_names[pp]}_{os.path.basename(img_path)}"
                    )

    print(f"\n📊 {name} Classification Report\n")
    print(classification_report(targets, preds, target_names=class_names))

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"{name} Confusion Matrix")
    writer.add_figure(f"{name}/ConfusionMatrix", plt.gcf())
    plt.show()

    for idx in [5, 20, 50]:
        save_gradcam(model, name, idx)

# =========================================================
# 8. MAIN
# =========================================================
def main():
    models_to_eval = {
        "VAF": VAF(num_classes),
        "UNET": UNET(num_classes),
        "CustomCNN": CustomCNN(num_classes),
        "ResNet50": build_resnet50(num_classes),
        "ResNet50_finetuned": build_resnet50(num_classes),
    }

    for name, model in models_to_eval.items():
        model_path = f"{MODEL_DIR}/{name}.pt"
        if not os.path.exists(model_path):
            print(f"⚠ Skipping {name}")
            continue

        print(f"\n===== EVALUATING {name} =====")
        writer = SummaryWriter(f"{TB_DIR}/{name}")

        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)

        evaluate_model(model, name, writer)
        writer.close()

    print("\n✅ EVALUATION COMPLETE")

# =========================================================
# 9. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
