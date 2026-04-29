import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from autoencoder import build_autoencoder


# ─────────────────────────────────────────
#  HYPERPARAMETERS
#  These are the knobs you can tune
# ─────────────────────────────────────────
LATENT_DIM   = 128      # size of bottleneck (128 numbers to represent an image)
EPOCHS       = 50       # how many times to loop through entire dataset
BATCH_SIZE   = 64       # how many images per gradient update
LEARNING_RATE = 1e-3    # how big each weight update step is
SAVE_DIR     = "saved_model"


# ─────────────────────────────────────────
#  CALLBACKS
#  These run automatically during training
#  to help you monitor and control it
# ─────────────────────────────────────────
def get_callbacks():
    os.makedirs(SAVE_DIR, exist_ok=True)

    callbacks = [

        # 1. Save best model automatically
        #    Watches val_loss — if it improves, saves weights
        #    So even if training crashes at epoch 48, you have epoch 40's best weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "autoencoder_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # 2. Early stopping
        #    If val_loss doesn't improve for 7 epochs in a row → stop training
        #    Prevents wasting time and overfitting
        #    restore_best_weights=True → snaps back to the best epoch automatically
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        # 3. Learning rate reducer
        #    If val_loss plateaus for 3 epochs → divide learning rate by 2
        #    Helps the model fine-tune in the later stages of training
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        # 4. TensorBoard logging (optional but cool)
        #    Run: tensorboard --logdir logs/
        #    Then open http://localhost:6006 to watch training live
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=1
        )
    ]

    return callbacks


# ─────────────────────────────────────────
#  PLOT TRAINING CURVES
#  Always visualize your training —
#  it tells you if something is wrong
# ─────────────────────────────────────────
def plot_training(history):
    """
    Plots train loss vs validation loss over epochs.

    What to look for:
        Good:  both curves going down together
        Bad:   train loss down, val loss going UP → overfitting
        Bad:   both curves barely moving → learning rate too low
    """
    plt.figure(figsize=(10, 4))

    plt.plot(history.history['loss'],     label='Train Loss',      color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("Saved: training_curve.png")


# ─────────────────────────────────────────
#  VISUALIZE RECONSTRUCTIONS
#  After training, run some images through
#  and see how well it rebuilds them
# ─────────────────────────────────────────
def visualize_reconstructions(autoencoder, x_samples, n=8):
    """
    Shows original vs reconstructed images side by side.
    Good reconstructions = model learned clean image structure.
    """
    reconstructed = autoencoder.predict(x_samples[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        # Top row: originals
        axes[0, i].imshow(x_samples[i])
        axes[0, i].axis('off')
        axes[0, i].set_title("Original", fontsize=7)

        # Bottom row: reconstructions
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        axes[1, i].set_title("Rebuilt", fontsize=7)

    plt.suptitle("Original vs Reconstructed (clean images)", fontsize=12)
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.show()
    print("Saved: reconstructions.png")


# ─────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────
def train():
    # ── Load data ──────────────────────────
    print("Loading data...")
    x_train_clean = np.load("x_train_clean.npy")   # shape: (50000, 32, 32, 3)
    print(f"Training data shape: {x_train_clean.shape}")

    # ── Build model ────────────────────────
    print("\nBuilding autoencoder...")
    autoencoder, encoder, decoder = build_autoencoder(latent_dim=LATENT_DIM)
    print(f"Total parameters: {autoencoder.count_params():,}")

    # ── GPU check ──────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU detected: {gpus[0].name} ✓")
    else:
        print("\nNo GPU detected — running on CPU (will be slower)")

    # ── Train ──────────────────────────────
    # Note: x = y = x_train_clean
    # We're teaching it: "given this image, output this same image"
    # The model learns to compress + reconstruct clean images
    print(f"\nStarting training — {EPOCHS} epochs, batch size {BATCH_SIZE}...")
    history = autoencoder.fit(
        x_train_clean,                          # input:  clean images
        x_train_clean,                          # target: same clean images
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,                           # shuffle each epoch
        validation_split=0.1,                   # use 10% of train data as validation
        callbacks=get_callbacks(),
        verbose=1
    )

    # ── Save final model ───────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    autoencoder.save(os.path.join(SAVE_DIR, "autoencoder_final.keras"))
    encoder.save(os.path.join(SAVE_DIR, "encoder.keras"))
    decoder.save(os.path.join(SAVE_DIR, "decoder.keras"))
    print(f"\nModels saved to '{SAVE_DIR}/'")

    # ── Plots ──────────────────────────────
    plot_training(history)
    visualize_reconstructions(autoencoder, x_train_clean)

    # ── Final loss ─────────────────────────
    final_train_loss = history.history['loss'][-1]
    final_val_loss   = history.history['val_loss'][-1]
    print(f"\nFinal train loss: {final_train_loss:.6f}")
    print(f"Final val loss:   {final_val_loss:.6f}")
    print("\nTraining complete!")

    return autoencoder, encoder, decoder, history


# ─────────────────────────────────────────
#  Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    autoencoder, encoder, decoder, history = train()
