import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = 96          
BATCH_SIZE = 64
EPOCHS = 30            
LR = 1e-3
VAL_FRACTION = 0.10    
OUT_DIR = os.path.join("results", "q5c")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

# ---------------------------
#         Data
# ---------------------------
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=VAL_FRACTION,
    stratify=y_train_full, random_state=SEED)

n_total = len(x_train_full) + len(x_test)
split_table = {
    "Totalt": n_total,
    "Trening": len(x_train),
    "Validering": len(x_val),
    "Test": len(x_test),
}
print("\nDatasett-oppdeling:")
for k, v in split_table.items():
    print(f"  {k:<11} {v:>6}  ({100 * v / n_total:5.1f} %)")

# Made by AI. It only partly shuffles to save memory andtime 
def make_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(10_000, seed=SEED)

    def _prep(img, label):
        img = tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE))
        if training:
            # Reccomended from claude to avaid overfitting. It rotates the image and flips it horizontally to create more training data.
            img = tf.image.random_flip_left_right(img)
        return img, label

    return (ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)
test_ds = make_ds(x_test, y_test, training=False)

# ----------------------
#       Model 
# ----------------------
def build_transfer_learning_model():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    # Model 1: MobileNetV2
    x1 = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    mobilenet = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    mobilenet._name = "mobilenetv2_backbone"
    mobilenet.trainable = False                      
    x = mobilenet(x1, training=False)               
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_mobilenet")(x)  # 1280
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)

    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    out = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax",
                                name="output")(x)           

    return tf.keras.Model(inp, out, name="transfer_learning_mobilenetv2")

model = build_transfer_learning_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss="sparse_categorical_crossentropy",      
              metrics=["accuracy"])
model.summary()

trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
total_params = int(model.count_params())
print(f"\nParametere: totalt {total_params:,} | trenbare {trainable_params:,}")

# -----------------
#      Training
# ---------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=2, min_lr=1e-6, verbose=1),
]

t0 = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=callbacks, verbose=1)
train_time = time.time() - t0
epochs_run = len(history.history["loss"])
print(f"\nTreningstid: {train_time:.1f} s over {epochs_run} epoker")

# ----------------------
#       loss curve
# ----------------------
h = history.history
ep = np.arange(1, epochs_run + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ep, h["loss"], "o-", label="Train loss")
ax[0].plot(ep, h["val_loss"], "s-", label="Val loss")
ax[0].set_xlabel("Epoke"); ax[0].set_ylabel("Loss"); ax[0].set_title("Tap")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ep, h["accuracy"], "o-", label="Train accuracy")
ax[1].plot(ep, h["val_accuracy"], "s-", label="Val accuracy")
ax[1].set_xlabel("Epoke"); ax[1].set_ylabel("Accuracy"); ax[1].set_title("Nøyaktighet")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Q5(a) Fusion (MobileNetV2 + ResNet50) – trening")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_accuracy_curves.png"), dpi=150)

gap = h["val_loss"][-1] - h["loss"][-1]
print(f"Siste epoke: train loss {h['loss'][-1]:.3f}, val loss {h['val_loss'][-1]:.3f} "
      f"(gap {gap:+.3f})")

