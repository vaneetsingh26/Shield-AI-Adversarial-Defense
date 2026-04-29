import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

# ─────────────────────────────────────────
#  STEP 1 — Load CIFAR-10
# ─────────────────────────────────────────
def load_data():
    """
    Loads CIFAR-10 and normalizes pixel values from [0, 255] to [0.0, 1.0]
    Neural networks work much better with small numbers.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    print(f"Train set: {x_train.shape}")   # (50000, 32, 32, 3)
    print(f"Test set:  {x_test.shape}")    # (10000, 32, 32, 3)

    return x_train, y_train, x_test, y_test


# ─────────────────────────────────────────
#  STEP 2 — Build a simple CNN
#  (We only need this to generate gradients
#   for FGSM. It's NOT our detector.)
# ─────────────────────────────────────────
def build_classifier():
    """
    Small CNN trained on CIFAR-10.
    We use this ONLY to compute FGSM attack gradients.
    Think of it as the 'victim model' we're fooling.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')   # 10 CIFAR classes
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_classifier(model, x_train, y_train, x_test, y_test):
    """
    Train the victim CNN quickly (just 5 epochs, we don't need it perfect).
    We just need it good enough to have meaningful gradients.
    """
    print("\nTraining victim classifier (for FGSM only)...")
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )
    return model


# ─────────────────────────────────────────
#  STEP 3 — FGSM Attack
#  This is the actual poisoning logic
# ─────────────────────────────────────────
def fgsm_attack(model, images, labels, epsilon=0.03):
    """
    FGSM = Fast Gradient Sign Method

    The math:
        x_poisoned = x + epsilon * sign( gradient of loss w.r.t. x )

    What this means in plain English:
        - We ask: "which direction should I nudge each pixel to maximize the loss?"
        - We nudge every pixel just a tiny bit (epsilon = 0.03) in that direction
        - The image looks identical to humans but fools classifiers

    Args:
        model   : the victim CNN
        images  : clean images  shape (N, 32, 32, 3)
        labels  : true labels   shape (N, 1)
        epsilon : how strong the perturbation is (0.03 = invisible)

    Returns:
        poisoned_images : same shape as images
    """
    images_tensor = tf.cast(images, tf.float32)
    labels_tensor = tf.cast(labels, tf.int64)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # GradientTape watches operations so we can differentiate them
    with tf.GradientTape() as tape:
        tape.watch(images_tensor)                          # watch the INPUT (not weights)
        predictions = model(images_tensor, training=False)
        loss = loss_fn(labels_tensor, predictions)

    # Get gradient of loss with respect to each pixel
    gradients = tape.gradient(loss, images_tensor)        # shape: (N, 32, 32, 3)

    # sign() converts gradient to -1 or +1 per pixel
    signed_gradients = tf.sign(gradients)

    # Nudge pixels
    poisoned = images_tensor + epsilon * signed_gradients

    # Clip to keep pixel values valid [0, 1]
    poisoned = tf.clip_by_value(poisoned, 0.0, 1.0)

    return poisoned.numpy()


# ─────────────────────────────────────────
#  STEP 4 — Build Full Dataset
#  Clean images (label=0) + Poisoned (label=1)
# ─────────────────────────────────────────
def build_detection_dataset(x_train, y_train, x_test, y_test):
    """
    Creates a balanced dataset for our autoencoder:
        - Autoencoder trains ONLY on clean images (no labels needed)
        - For testing, we have clean + poisoned with binary labels

    Returns:
        x_train_clean   : clean training images (for autoencoder training)
        x_test_images   : mixed clean + poisoned test images
        x_test_labels   : 0 = clean, 1 = poisoned
    """
    print("\nBuilding victim classifier...")
    classifier = build_classifier()
    classifier = train_classifier(classifier, x_train, y_train, x_test, y_test)

    # Poison the test set
    print("\nGenerating FGSM poisoned images...")
    x_test_poisoned = fgsm_attack(classifier, x_test, y_test, epsilon=0.03)
    print(f"Poisoned images shape: {x_test_poisoned.shape}")

    # Mix clean + poisoned for test set
    x_test_mixed  = np.concatenate([x_test, x_test_poisoned], axis=0)
    y_test_mixed  = np.array([0] * len(x_test) + [1] * len(x_test_poisoned))

    # Shuffle the mixed test set
    idx = np.random.permutation(len(x_test_mixed))
    x_test_mixed = x_test_mixed[idx]
    y_test_mixed = y_test_mixed[idx]

    print(f"\nDataset ready:")
    print(f"  Autoencoder train (clean only): {x_train.shape}")
    print(f"  Detection test set: {x_test_mixed.shape}")
    print(f"  Test labels: {y_test_mixed.shape}  (0=clean, 1=poisoned)")

    return x_train, x_test_mixed, y_test_mixed, classifier


# ─────────────────────────────────────────
#  Run this file directly to test it
# ─────────────────────────────────────────
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    x_train_clean, x_test_mixed, y_test_labels, clf = build_detection_dataset(
        x_train, y_train, x_test, y_test
    )
    print("\ndata_loader.py working correctly!")

    # Save for use in other files
    np.save("x_train_clean.npy", x_train_clean)
    np.save("x_test_mixed.npy",  x_test_mixed)
    np.save("y_test_labels.npy", y_test_labels)
    print("Saved: x_train_clean.npy, x_test_mixed.npy, y_test_labels.npy")
