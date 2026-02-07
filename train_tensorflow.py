# =========================================================
# 0. SYSTEM SETUP
# =========================================================



import os, json, csv
import tensorflow as tf
from tensorflow.keras import layers, Model

# Performance (safe for Colab)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

print("GPU:", tf.config.list_physical_devices('GPU'))

# =========================================================
# 1. CONFIG
# =========================================================

DATASET_DIR = "./Brain Cancer"
CHECKPOINT_DIR = "./checkpoints"
MODEL_DIR = "./trained_models"
LOG_DIR = "./logs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS_BASE = 30
EPOCHS_FINE = 10
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

for d in [CHECKPOINT_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 2. DATASET
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

NUM_CLASSES = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# =========================================================
# 3. FAST PIPELINE (NO CACHE)
# =========================================================

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def prepare(ds, training=False):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, True)
val_ds = prepare(val_ds, False)

# =========================================================
# 4. MODELS
# =========================================================

def build_vaf():
    inp = layers.Input((*IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, strides=2, activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="VAF")

def build_unet():
    inp = layers.Input((*IMG_SIZE, 3))
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="UNET")

def build_custom_cnn():
    return tf.keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")
    ], name="CustomCNN")

from tensorflow.keras.applications import ResNet50

def build_resnet50():
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inp = layers.Input((*IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="ResNet50")

# =========================================================
# 5. RESUME UTILITIES
# =========================================================

def get_last_epoch(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path) as f:
        rows = list(csv.reader(f))
        return max(0, len(rows) - 1)

# =========================================================
# 6. TRAIN (EPOCH-ACCURATE, CRASH-PROOF)
# =========================================================

def train_model(model, name, epochs, lr):
    print(f"\n===== TRAINING {name} =====")

    ckpt_dir = os.path.join(CHECKPOINT_DIR, name)
    log_dir = os.path.join(LOG_DIR, name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_best = os.path.join(ckpt_dir, "best.weights.h5")
    ckpt_last = os.path.join(ckpt_dir, "last.weights.h5")
    csv_log = os.path.join(log_dir, "training_log.csv")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Load LAST checkpoint for exact resume
    if os.path.exists(ckpt_last):
        model.load_weights(ckpt_last)
        print("✔ Loaded last epoch checkpoint")
    elif os.path.exists(ckpt_best):
        model.load_weights(ckpt_best)
        print("✔ Loaded best checkpoint")

    initial_epoch = get_last_epoch(csv_log)
    print(f"✔ Resuming from epoch {initial_epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_best,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_last,
            save_best_only=False,
            save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(csv_log, append=True),
        tf.keras.callbacks.EarlyStopping(patience=6),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, name))
    return model

# =========================================================
# 7. RESNET FINE-TUNING
# =========================================================

def fine_tune_resnet(model, unfreeze_layers=30):
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break

    for layer in base.layers[-unfreeze_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================================================
# 8. RUN ALL MODELS
# =========================================================

models = {
    "VAF": build_vaf(),
    "UNET": build_unet(),
    "CustomCNN": build_custom_cnn(),
    "ResNet50": build_resnet50()
}

for name, model in models.items():
    model = train_model(model, name, EPOCHS_BASE, 3e-4)

    if name == "ResNet50":
        print("\n--- Fine-tuning ResNet50 ---")
        model = fine_tune_resnet(model)
        train_model(model, name + "_finetuned", EPOCHS_FINE, 1e-5)

print("\n✅ ALL MODELS TRAINED & RESUMABLE")

