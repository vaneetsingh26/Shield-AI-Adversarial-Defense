import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# ─────────────────────────────────────────
#  CONCEPT REMINDER
#
#  After training on clean images only:
#    Clean image  → autoencoder rebuilds it well  → LOW error
#    Poisoned image → autoencoder strips the noise → HIGH error
#
#  We just need to find the right threshold:
#    error < threshold → CLEAN
#    error > threshold → POISONED
# ─────────────────────────────────────────


class PoisonDetector:

    def __init__(self, model_path="saved_model/autoencoder_best.keras"):
        """
        Loads the trained autoencoder.
        threshold is None until you call find_threshold()
        """
        print(f"Loading model from {model_path}...")
        self.autoencoder = tf.keras.models.load_model(model_path)
        self.threshold   = None
        print("Model loaded ✓")


    # ─────────────────────────────────────
    #  RECONSTRUCTION ERROR
    #  Core of the whole detector
    # ─────────────────────────────────────
    def reconstruction_error(self, images):
        """
        Runs images through autoencoder and computes
        per-image Mean Squared Error between original and rebuild.

        Args:
            images: np array of shape (N, 32, 32, 3), values in [0, 1]

        Returns:
            errors: np array of shape (N,) — one error score per image
                    higher score = more likely poisoned
        """
        reconstructed = self.autoencoder.predict(images, verbose=0)

        # MSE per image: mean over all pixels and channels
        # axis=(1,2,3) means average over height, width, channels
        errors = np.mean(np.square(images - reconstructed), axis=(1, 2, 3))

        return errors


    # ─────────────────────────────────────
    #  FIND THRESHOLD
    #
    #  Strategy: run all clean test images through
    #  Take the 95th percentile of their errors
    #  → 95% of clean images will be below this line
    #  → poisoned images (with higher errors) will be above it
    # ─────────────────────────────────────
    def find_threshold(self, x_clean, percentile=95):
        """
        Computes the detection threshold from clean images.

        Why 95th percentile?
            - Too low (e.g. 50th) → too many clean images flagged as poisoned (false positives)
            - Too high (e.g. 99th) → miss some poisoned images (false negatives)
            - 95th is a good balance

        You can tune this based on your use case.

        Args:
            x_clean    : clean images only, shape (N, 32, 32, 3)
            percentile : 95 by default

        Returns:
            threshold  : float
        """
        print(f"Computing threshold from {len(x_clean)} clean images...")
        errors = self.reconstruction_error(x_clean)

        self.threshold = float(np.percentile(errors, percentile))

        print(f"Clean image errors — min: {errors.min():.6f}, "
              f"max: {errors.max():.6f}, mean: {errors.mean():.6f}")
        print(f"Threshold ({percentile}th percentile): {self.threshold:.6f}")

        # Save threshold to disk so you don't recompute every time
        np.save("saved_model/threshold.npy", np.array([self.threshold]))
        print("Threshold saved to saved_model/threshold.npy")

        return self.threshold


    def load_threshold(self, path="saved_model/threshold.npy"):
        """Load a previously computed threshold."""
        self.threshold = float(np.load(path)[0])
        print(f"Loaded threshold: {self.threshold:.6f}")
        return self.threshold


    # ─────────────────────────────────────
    #  DETECT — single image or batch
    # ─────────────────────────────────────
    def detect(self, images):
        """
        Main detection function.

        Args:
            images: shape (N, 32, 32, 3) or (32, 32, 3) for single image

        Returns:
            results: list of dicts with keys:
                       label      → 'CLEAN' or 'POISONED'
                       confidence → float 0-100 (how confident we are)
                       error      → raw reconstruction error score
        """
        assert self.threshold is not None, "Run find_threshold() or load_threshold() first!"

        # Handle single image
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        errors = self.reconstruction_error(images)

        results = []
        for error in errors:
            label = "POISONED" if error > self.threshold else "CLEAN"

            # Confidence: how far is this error from the threshold?
            # We map the distance to a 0-100 confidence score
            # The further above threshold → more confident it's poisoned
            # The further below threshold → more confident it's clean
            distance = abs(error - self.threshold)
            max_range = self.threshold * 3               # rough normalization range
            raw_conf  = min(distance / max_range, 1.0)  # clip to [0, 1]
            confidence = 50 + raw_conf * 50              # map to [50, 100]

            results.append({
                "label":      label,
                "confidence": round(confidence, 1),
                "error":      round(float(error), 6)
            })

        return results


    # ─────────────────────────────────────
    #  ERROR HEATMAP
    #
    #  Instead of just saying "poisoned",
    #  show WHERE the anomaly is in the image
    #
    #  How it works:
    #    Compute squared error per pixel → (32, 32, 3)
    #    Average across channels         → (32, 32)
    #    Apply colormap (red = high error)
    # ─────────────────────────────────────
    def get_error_heatmap(self, image):
        """
        Generates a heatmap showing where the reconstruction error is highest.
        Red areas = where the model sees anomalies = where the poison is.

        Args:
            image: single image, shape (32, 32, 3)

        Returns:
            heatmap_colored: RGB heatmap, shape (32, 32, 3), values [0, 1]
            error_map      : raw error per pixel, shape (32, 32)
        """
        img_batch     = np.expand_dims(image, axis=0)           # (1, 32, 32, 3)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)[0]  # (32, 32, 3)

        # Squared error per pixel, averaged across RGB channels
        error_map = np.mean(np.square(image - reconstructed), axis=-1)   # (32, 32)

        # Normalize to [0, 1] for colormap
        error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)

        # Apply 'hot' colormap: black → red → yellow → white
        colormap       = cm.get_cmap('hot')
        heatmap_colored = colormap(error_norm)[:, :, :3]         # drop alpha channel

        return heatmap_colored, error_map


    # ─────────────────────────────────────
    #  EVALUATE ON TEST SET
    #  Run detector on mixed clean+poisoned
    #  and compute accuracy, precision, recall
    # ─────────────────────────────────────
    def evaluate(self, x_test_mixed, y_test_labels):
        """
        Evaluates detector performance on the test set.

        Args:
            x_test_mixed  : shape (N, 32, 32, 3)  — mix of clean and poisoned
            y_test_labels : shape (N,)             — 0=clean, 1=poisoned

        Prints accuracy, precision, recall, F1
        """
        assert self.threshold is not None, "Run find_threshold() first!"

        print(f"\nEvaluating on {len(x_test_mixed)} images...")
        errors     = self.reconstruction_error(x_test_mixed)
        predictions = (errors > self.threshold).astype(int)   # 1 if poisoned, 0 if clean

        # Confusion matrix components
        TP = np.sum((predictions == 1) & (y_test_labels == 1))  # correctly flagged poisoned
        TN = np.sum((predictions == 0) & (y_test_labels == 0))  # correctly flagged clean
        FP = np.sum((predictions == 1) & (y_test_labels == 0))  # clean flagged as poisoned
        FN = np.sum((predictions == 0) & (y_test_labels == 1))  # poisoned missed

        accuracy  = (TP + TN) / len(y_test_labels) * 100
        precision = TP / (TP + FP + 1e-8) * 100
        recall    = TP / (TP + FN + 1e-8) * 100
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"\n{'='*40}")
        print(f"  DETECTION RESULTS")
        print(f"{'='*40}")
        print(f"  Accuracy:  {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall:    {recall:.2f}%")
        print(f"  F1 Score:  {f1:.2f}%")
        print(f"{'='*40}")
        print(f"  TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
        print(f"{'='*40}\n")

        return {"accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1}


    # ─────────────────────────────────────
    #  VISUALIZE DETECTIONS
    #  Show a grid of images with their
    #  detection results and heatmaps
    # ─────────────────────────────────────
    def visualize_detections(self, x_images, y_true=None, n=6):
        """
        Displays n images with:
            - Original image
            - Heatmap overlay
            - Predicted label + confidence
            - True label (if provided)
        """
        results = self.detect(x_images[:n])

        fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))

        for i in range(n):
            heatmap, _ = self.get_error_heatmap(x_images[i])
            r          = results[i]

            # Color border: green = clean, red = poisoned
            border_color = 'red' if r['label'] == 'POISONED' else 'green'

            # Top row: original
            axes[0, i].imshow(x_images[i])
            axes[0, i].axis('off')
            title = f"{r['label']}\n{r['confidence']}%"
            if y_true is not None:
                true_str = "poisoned" if y_true[i] == 1 else "clean"
                title += f"\n(true: {true_str})"
            axes[0, i].set_title(title, fontsize=7, color=border_color)

            # Bottom row: heatmap
            axes[1, i].imshow(heatmap)
            axes[1, i].axis('off')
            axes[1, i].set_title("Error map", fontsize=7)

        plt.suptitle("Detection Results  |  Top: Original  |  Bottom: Error Heatmap", fontsize=10)
        plt.tight_layout()
        plt.savefig("detection_results.png", dpi=150)
        plt.show()
        print("Saved: detection_results.png")


# ─────────────────────────────────────────
#  Run this file directly to test detector
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    x_train_clean = np.load("x_train_clean.npy")
    x_test_mixed  = np.load("x_test_mixed.npy")
    y_test_labels = np.load("y_test_labels.npy")

    # Build detector
    detector = PoisonDetector("saved_model/autoencoder_best.keras")

    # Set percentile lower (e.g., 60) to make the detector aggressively strict
    detector.find_threshold(x_train_clean[:5000], percentile=60)

    # Evaluate on test set
    metrics = detector.evaluate(x_test_mixed, y_test_labels)

    # Visualize some detections
    detector.visualize_detections(x_test_mixed, y_test_labels, n=6)

    # Test single image detection
    print("\nSingle image detection test:")
    single = x_test_mixed[0:1]
    result = detector.detect(single)
    print(f"  Result: {result[0]}")
