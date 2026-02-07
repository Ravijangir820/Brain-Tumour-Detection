# =========================================================
# 0. SYSTEM SETUP (WINDOWS SAFE)
# =========================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 1. IMPORTS
# =========================================================
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# =========================================================
# 2. CONFIG
# =========================================================
DATASET_DIR = "./Brain Cancer"

LOG_DIR = "./logs"
CKPT_DIR = "./checkpoints"
MODEL_DIR = "./trained_models"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 30
FINETUNE_EPOCHS = 10
LR = 3e-4
FINETUNE_LR = 1e-5
PATIENCE = 6
NUM_WORKERS = 4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in [LOG_DIR, CKPT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 3. REPRODUCIBILITY
# =========================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =========================================================
# 4. TRANSFORMS (TF-EQUIVALENT)
# =========================================================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# 5. DATASET & LOADERS
# =========================================================
full_ds = datasets.ImageFolder(DATASET_DIR)
num_classes = len(full_ds.classes)

indices = np.arange(len(full_ds))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx, val_idx = indices[:split], indices[split:]

train_ds = Subset(datasets.ImageFolder(DATASET_DIR, transform=train_tfms), train_idx)
val_ds   = Subset(datasets.ImageFolder(DATASET_DIR, transform=val_tfms), val_idx)

train_loader = DataLoader(
    train_ds, BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

val_loader = DataLoader(
    val_ds, BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

# =========================================================
# 6. MODEL DEFINITIONS
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
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n)
    return model

# =========================================================
# 7. GENERIC TRAIN FUNCTION (LOSS + ACC LOGGING)
# =========================================================
def train_model(model, name, epochs, lr):
    print(f"\n===== TRAINING {name} =====")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    scaler = GradScaler("cuda")

    csv_path = f"{LOG_DIR}/{name}.csv"
    ckpt_path = f"{CKPT_DIR}/{name}_best.pt"

    best_acc = 0.0
    wait = 0

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        correct = total = 0
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"{name} | Epoch {epoch+1}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # -------- VALIDATE --------
        model.eval()
        correct = total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, train_loss, train_acc, val_loss, val_acc, lr_now]
            )

        print(
            f"{name} | Epoch {epoch+1}: "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} LR={lr_now:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"⛔ Early stopping {name}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    torch.save(model.state_dict(), f"{MODEL_DIR}/{name}.pt")
    return model

# =========================================================
# 8. RESNET FINETUNING (TF-STYLE)
# =========================================================
def finetune_resnet(model):
    print("\n🔬 Fine-tuning ResNet50")

    for p in model.parameters():
        p.requires_grad = False

    backbone = list(model.children())[0]
    for layer in list(backbone.children())[-30:]:
        for p in layer.parameters():
            p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True

    return train_model(
        model,
        "ResNet50_finetuned",
        FINETUNE_EPOCHS,
        FINETUNE_LR
    )

# =========================================================
# 9. MAIN
# =========================================================
def main():
    models_to_train = {
        "VAF": VAF(num_classes),
        "UNET": UNET(num_classes),
        "CustomCNN": CustomCNN(num_classes),
        "ResNet50": build_resnet50(num_classes)
    }

    trained = {}

    for name, model in models_to_train.items():
        trained[name] = train_model(model, name, EPOCHS, LR)

    trained["ResNet50"] = finetune_resnet(trained["ResNet50"])

    print("\n✅ TRAINING COMPLETE — LOGS, CHECKPOINTS & MODELS SAVED")

# =========================================================
# 10. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
