# MAS512 - Q4 Classification
# Classify the state of health of a motor: 00_15 (Healthy) / 10_15 (Faulty)
# Data loading, baseline model and training settings follow the lab notebook from class.
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)

DATA_DIR = os.path.join("Assignment 1", "Assignment 1-class", "fault-classification-class")
n_row = 60
n_col = 175
image_size = (n_row, n_col)
batch_size = 8
EPOCHS = 15

 # Data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "training"),
    labels='inferred',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=120,
    validation_split=0.2,   
    subset='training',
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "training"),
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    seed=120,
    validation_split=0.2,  
    subset='validation',
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "testing"),   
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    seed=120,
)

class_names = train_ds.class_names
print("The classes are", class_names)


# Baseline: model from the lab notebook
normalizer = tf.keras.layers.Normalization(axis=-1) 
normalizer.adapt(train_ds.map(lambda x, y: x))

baseline = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_row, n_col, 3)),
    normalizer,
    tf.keras.layers.Conv2D(4, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activity_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'),
    tf.keras.layers.Dense(16, activity_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(64, activity_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'),
    tf.keras.layers.Dense(64, activity_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax'),
], name="baseline")

# our model: 
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_row, n_col, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),              
    tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation="softmax"),
], name="my_model")


# Train 
y_true = np.concatenate([y for x, y in test_ds], axis=0)
results = {}

for model in [baseline, my_model]:
    name = model.name
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"],)
    history = model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds)

    # (b) 
    plt.figure(figsize=(5, 5))
    plt.plot(range(EPOCHS), history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(range(EPOCHS), history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.legend(loc='upper right')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(name)
    plt.savefig(f"q4_{name}_loss.png", dpi=150)

    start = time.time()
    y_prob = model.predict(test_ds)
    inference_time = (time.time() - start) / len(y_true) * 1000   # ms per image
    y_pred = np.argmax(y_prob, axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    sn.heatmap(cm_n, xticklabels=['Healthy', 'Faulty'], yticklabels=['Healthy', 'Faulty'], annot=True, fmt='.2f', annot_kws={"size": 16})
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(name)
    plt.savefig(f"q4_{name}_confusion_matrix.png", dpi=150)

    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred))

    # (c) 
    results[name] = [model.count_params(),
                     inference_time,
                     accuracy_score(y_true, y_pred),
                     precision_score(y_true, y_pred),
                     recall_score(y_true, y_pred),
                     f1_score(y_true, y_pred)]

print("\nModel        Parameters  Inference [ms]  Accuracy  Precision  Recall     F1")
for name, r in results.items():
    print(f"{name:<12} {r[0]:>10} {r[1]:>15.3f} {r[2]:>9.3f} {r[3]:>10.3f} {r[4]:>7.3f} {r[5]:>6.3f}")

plt.show()