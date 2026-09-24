import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = os.path.join("Assignment 1/Assignment 1-class/fault-classification-class")
IMG_SIZE   = 64      
LATENT_DIM = 32      
EPOCHS     = 50
BATCH_SIZE = 32
N_SHOW     = 10      

input_dim = IMG_SIZE * IMG_SIZE

def load_folder(path):
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale", # reccomended by claude ai to use grayscale images for autoencoders to reduce complexity and focus on structural features
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        shuffle=False,
    )
    X, y = [], []
    for img, lab in ds:
        X.append(img.numpy())
        y.append(lab.numpy())
    X = np.concatenate(X) / 255.0           
    y = np.concatenate(y)
    X = X.reshape(len(X), input_dim)         # flatten: 64x64 -> 4096
    return X, y, ds.class_names

X_train, y_train, class_names = load_folder(os.path.join(DATA_DIR, "training"))
X_test,  y_test,  _           = load_folder(os.path.join(DATA_DIR, "testing"))
print("Classes:", class_names)               
print("Train:", X_train.shape, " Test:", X_test.shape)

# Encoder: 
encoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(LATENT_DIM, activation="relu"),
], name="encoder")

# Decoder: 
decoder = keras.Sequential([
    layers.Input(shape=(LATENT_DIM,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid"),  # sigmoid to get an output in the range [0, 1], which is the same range as the input images
], name="decoder")

autoencoder = keras.Sequential([encoder, decoder], name="autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Training 
history = autoencoder.fit(X_train, X_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, X_test),
)

plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
plt.title("Autoencoder training")
plt.savefig("q6_loss_curve.png", dpi=150)

rng = np.random.default_rng(42)
mean_loss = {}

for c, name in enumerate(class_names):
    idx = np.where(y_test == c)[0]
    idx = rng.choice(idx, size=N_SHOW, replace=False)

    x = X_test[idx]                               # (a) originalbilder
    z = encoder.predict(x, verbose=0)             # (b) latent vektor
    x_hat = decoder.predict(z, verbose=0)         # (c) rekonstruksjon
    err = np.mean((x - x_hat) ** 2, axis=1)       # (d) MSE per bilde
    mean_loss[name] = err.mean()


    n = len(idx)
    fig, ax = plt.subplots(3, n, figsize=(2 * n, 6.5), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(x[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax[0, i].set_title(f"#{i+1}")
        ax[1, i].imshow(z[i].reshape(4, LATENT_DIM // 4), cmap="viridis")
        ax[2, i].imshow(x_hat[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax[2, i].set_title(f"MSE={err[i]:.4f}", fontsize=8)
        for r in range(3):
            ax[r, i].axis("off")
    ax[0, 0].text(-0.3, 0.5, "(a) Original", transform=ax[0, 0].transAxes,
                  rotation=90, va="center", ha="right")
    ax[1, 0].text(-0.3, 0.5, f"(b) Latent ({LATENT_DIM})", transform=ax[1, 0].transAxes,
                  rotation=90, va="center", ha="right")
    ax[2, 0].text(-0.3, 0.5, "(c) Reconstructed", transform=ax[2, 0].transAxes,
                  rotation=90, va="center", ha="right")
    fig.suptitle(f"Class: {name}   |   mean MSE = {mean_loss[name]:.5f}")
    plt.tight_layout()
    plt.savefig(f"q6_class_{name}.png", dpi=150)

print("\n===== Average reconstruction loss per class =====")
for name, l in mean_loss.items():
    print(f"{name:20s}: {l:.5f}")

plt.figure()
plt.bar(list(mean_loss.keys()), list(mean_loss.values()))
plt.ylabel("Mean MSE"); plt.title("Mean reconstruction loss per class")
plt.savefig("q6_mean_loss_per_class.png", dpi=150)
plt.show()