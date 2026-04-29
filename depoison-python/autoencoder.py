import tensorflow as tf
from keras import layers, Model


# ─────────────────────────────────────────
#  QUICK CONCEPT REMINDER
#
#  Encoder: Image → small number (compress)
#  Decoder: small number → Image (rebuild)
#
#  Conv2D        → scans image, extracts features, shrinks size
#  ConvTranspose → reverse of Conv2D, grows size back up
#  BatchNorm     → keeps numbers stable during training
#  ReLU          → activation, just kills negative values (max(0, x))
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  ENCODER
#  Takes (32, 32, 3) → outputs (128,)
#
#  Each Conv2D + Pool halves the spatial size:
#    32x32 → 16x16 → 8x8 → 4x4
#  Then we flatten and compress to 128 numbers
# ─────────────────────────────────────────
def build_encoder(latent_dim=128):
    inputs = layers.Input(shape=(32, 32, 3), name="encoder_input")

    # Block 1: (32, 32, 3) → (16, 16, 32)
    x = layers.Conv2D(32, (3, 3), padding='same', name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.ReLU(name="enc_relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool1")(x)          # halves to 16x16

    # Block 2: (16, 16, 32) → (8, 8, 64)
    x = layers.Conv2D(64, (3, 3), padding='same', name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.ReLU(name="enc_relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool2")(x)          # halves to 8x8

    # Block 3: (8, 8, 64) → (4, 4, 128)
    x = layers.Conv2D(128, (3, 3), padding='same', name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.ReLU(name="enc_relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool3")(x)          # halves to 4x4

    # Flatten: (4, 4, 128) → (2048,)
    x = layers.Flatten(name="enc_flatten")(x)

    # Compress to latent dim: (2048,) → (128,)
    # This is the BOTTLENECK — the model must summarize the entire image in 128 numbers
    latent = layers.Dense(latent_dim, activation='relu', name="latent")(x)

    encoder = Model(inputs, latent, name="Encoder")
    return encoder


# ─────────────────────────────────────────
#  DECODER
#  Takes (128,) → outputs (32, 32, 3)
#
#  Mirror image of encoder:
#    128 → reshape to 4x4 → 8x8 → 16x16 → 32x32
#  ConvTranspose2D = "reverse conv", upsamples
# ─────────────────────────────────────────
def build_decoder(latent_dim=128):
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")

    # Expand back from 128 → 4*4*128 = 2048
    x = layers.Dense(4 * 4 * 128, activation='relu', name="dec_dense")(inputs)

    # Reshape into spatial feature map: (2048,) → (4, 4, 128)
    x = layers.Reshape((4, 4, 128), name="dec_reshape")(x)

    # Block 1: (4, 4, 128) → (8, 8, 64)
    # Conv2DTranspose is like Conv2D but it UPSAMPLES (doubles size)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name="dec_convT1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.ReLU(name="dec_relu1")(x)

    # Block 2: (8, 8, 64) → (16, 16, 32)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name="dec_convT2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.ReLU(name="dec_relu2")(x)

    # Block 3: (16, 16, 32) → (32, 32, 3)
    # Sigmoid at end → squishes output back to [0, 1] range (same as our normalized images)
    outputs = layers.Conv2DTranspose(3, (3, 3), strides=2, padding='same',
                                     activation='sigmoid', name="dec_output")(x)

    decoder = Model(inputs, outputs, name="Decoder")
    return decoder


# ─────────────────────────────────────────
#  AUTOENCODER
#  Chains encoder + decoder together
#  Input: image → Output: reconstructed image
# ─────────────────────────────────────────
def build_autoencoder(latent_dim=128):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)

    # Connect them: image → encoder → latent → decoder → reconstructed image
    inputs       = layers.Input(shape=(32, 32, 3), name="autoencoder_input")
    latent       = encoder(inputs)
    reconstructed = decoder(latent)

    autoencoder = Model(inputs, reconstructed, name="Autoencoder")

    # Loss: MSE — how different is the reconstruction from the original?
    # Optimizer: Adam — standard, works well
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'                      # Mean Squared Error per pixel
    )

    return autoencoder, encoder, decoder


# ─────────────────────────────────────────
#  Run this file directly to preview shapes
# ─────────────────────────────────────────
if __name__ == "__main__":
    autoencoder, encoder, decoder = build_autoencoder(latent_dim=128)

    print("\n========= ENCODER =========")
    encoder.summary()

    print("\n========= DECODER =========")
    decoder.summary()

    print("\n========= FULL AUTOENCODER =========")
    autoencoder.summary()

    # Quick shape sanity check
    import numpy as np
    dummy_input = np.random.rand(1, 32, 32, 3).astype("float32")

    latent_output = encoder.predict(dummy_input)
    print(f"\nEncoder output shape: {latent_output.shape}")    # should be (1, 128)

    reconstructed = autoencoder.predict(dummy_input)
    print(f"Reconstructed shape:  {reconstructed.shape}")     # should be (1, 32, 32, 3)

    print("\nautoencoder.py working correctly!")
