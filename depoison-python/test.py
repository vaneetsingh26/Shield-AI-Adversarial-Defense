import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from detector import PoisonDetector


# ─────────────────────────────────────────
#  WHAT THIS FILE DOES
#
#  Ties everything together:
#  1. Loads test data
#  2. Runs detector on clean + poisoned images
#  3. Shows error score distributions
#  4. Shows side-by-side comparisons
#  5. Prints a final summary report
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  1. ERROR DISTRIBUTION PLOT
#
#  This is the most important visualization.
#  Shows reconstruction error for clean vs poisoned.
#  A good model will have two clearly separated peaks.
# ─────────────────────────────────────────
def plot_error_distribution(detector, x_clean_sample, x_poisoned_sample):
    """
    Plots histogram of reconstruction errors for clean vs poisoned images.

    What good looks like:
        Clean errors    → tight cluster on the LEFT  (low error)
        Poisoned errors → cluster shifted to the RIGHT (high error)
        Threshold line  → sits cleanly between the two peaks

    If the two clusters heavily overlap → model needs more training
    """
    print("Computing error distributions...")
    clean_errors    = detector.reconstruction_error(x_clean_sample)
    poisoned_errors = detector.reconstruction_error(x_poisoned_sample)

    plt.figure(figsize=(12, 5))

    # Plot clean error histogram
    plt.hist(clean_errors,    bins=60, alpha=0.6,
             color='green', label=f'Clean (n={len(x_clean_sample)})')

    # Plot poisoned error histogram
    plt.hist(poisoned_errors, bins=60, alpha=0.6,
             color='red',   label=f'Poisoned (n={len(x_poisoned_sample)})')

    # Draw threshold line
    if detector.threshold:
        plt.axvline(x=detector.threshold, color='black',
                    linestyle='--', linewidth=2,
                    label=f'Threshold = {detector.threshold:.4f}')

    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Number of Images')
    plt.title('Error Distribution: Clean vs Poisoned Images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=150)
    plt.show()
    print("Saved: error_distribution.png")

    # Print overlap stats
    overlap = np.sum(clean_errors > detector.threshold) / len(clean_errors) * 100
    print(f"False positive rate (clean above threshold): {overlap:.1f}%")

    return clean_errors, poisoned_errors


# ─────────────────────────────────────────
#  2. SIDE BY SIDE COMPARISON
#
#  For each pair: show clean vs poisoned version
#  of the SAME image with their error scores
#  Humans literally can't tell the difference —
#  but your model can
# ─────────────────────────────────────────
def plot_clean_vs_poisoned_pairs(detector, x_clean, x_poisoned, n=5):
    """
    Shows n pairs of clean vs poisoned versions of the same image.
    Demonstrates that the images look identical to humans
    but have very different reconstruction errors.
    """
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7))

    for i in range(n):
        clean_img    = x_clean[i]
        poisoned_img = x_poisoned[i]

        clean_result    = detector.detect(clean_img)[0]
        poisoned_result = detector.detect(poisoned_img)[0]

        clean_heatmap,    _ = detector.get_error_heatmap(clean_img)
        poisoned_heatmap,  _ = detector.get_error_heatmap(poisoned_img)

        # Row 1: Clean images
        axes[0, i].imshow(clean_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(
            f"CLEAN\nerr={clean_result['error']:.4f}",
            fontsize=7, color='green'
        )

        # Row 2: Poisoned images (look identical to row 1!)
        axes[1, i].imshow(poisoned_img)
        axes[1, i].axis('off')
        axes[1, i].set_title(
            f"POISONED\nerr={poisoned_result['error']:.4f}",
            fontsize=7, color='red'
        )

        # Row 3: Error heatmaps of poisoned images
        axes[2, i].imshow(poisoned_heatmap)
        axes[2, i].axis('off')
        axes[2, i].set_title("Heatmap", fontsize=7)

    axes[0, 0].set_ylabel("Clean",    fontsize=9, rotation=90)
    axes[1, 0].set_ylabel("Poisoned", fontsize=9, rotation=90)
    axes[2, 0].set_ylabel("Heatmap",  fontsize=9, rotation=90)

    plt.suptitle(
        "Clean vs Poisoned — Images look identical, errors are very different",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("clean_vs_poisoned.png", dpi=150)
    plt.show()
    print("Saved: clean_vs_poisoned.png")


# ─────────────────────────────────────────
#  3. EPSILON SENSITIVITY TEST
#
#  Tests how different epsilon values affect
#  detection. Higher epsilon = stronger poison
#  = easier to detect.
#  Lower epsilon = subtle poison = harder.
#  This shows your model's detection boundary.
# ─────────────────────────────────────────
def test_epsilon_sensitivity(detector, x_sample, y_sample, classifier):
    """
    Runs detection across multiple epsilon values.
    Shows how detection rate changes as poison gets weaker/stronger.

    Args:
        classifier : the victim CNN from data_loader (needed to generate FGSM)
    """
    from data_loader import fgsm_attack

    epsilons      = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    detection_rates = []

    print("\nTesting epsilon sensitivity...")
    for eps in epsilons:
        poisoned = fgsm_attack(classifier, x_sample, y_sample, epsilon=eps)
        results  = detector.detect(poisoned)
        detected = sum(1 for r in results if r['label'] == 'POISONED')
        rate     = detected / len(results) * 100
        detection_rates.append(rate)
        print(f"  ε={eps:.3f} → {rate:.1f}% detected")

    plt.figure(figsize=(9, 4))
    plt.plot(epsilons, detection_rates, marker='o', color='blue', linewidth=2)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    plt.xlabel('Epsilon (poison strength)')
    plt.ylabel('Detection Rate (%)')
    plt.title('Detection Rate vs Poison Strength (Epsilon)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("epsilon_sensitivity.png", dpi=150)
    plt.show()
    print("Saved: epsilon_sensitivity.png")

    return epsilons, detection_rates


# ─────────────────────────────────────────
#  4. FINAL REPORT
#  Clean summary of everything
# ─────────────────────────────────────────
def print_final_report(metrics, clean_errors, poisoned_errors, detector):
    sep = "=" * 50

    print(f"\n{sep}")
    print(f"   IMAGE POISON DETECTOR — FINAL REPORT")
    print(f"{sep}")
    print(f"  Model      : Convolutional Autoencoder")
    print(f"  Dataset    : CIFAR-10")
    print(f"  Attack     : FGSM (ε=0.03)")
    print(f"  Threshold  : {detector.threshold:.6f}")
    print(f"{sep}")
    print(f"  PERFORMANCE")
    print(f"  Accuracy   : {metrics['accuracy']:.2f}%")
    print(f"  Precision  : {metrics['precision']:.2f}%")
    print(f"  Recall     : {metrics['recall']:.2f}%")
    print(f"  F1 Score   : {metrics['f1']:.2f}%")
    print(f"{sep}")
    print(f"  ERROR STATS")
    print(f"  Clean    — mean: {clean_errors.mean():.6f}  std: {clean_errors.std():.6f}")
    print(f"  Poisoned — mean: {poisoned_errors.mean():.6f}  std: {poisoned_errors.std():.6f}")
    print(f"  Separation ratio: {poisoned_errors.mean() / clean_errors.mean():.2f}x")
    print(f"{sep}")
    print(f"  OUTPUT FILES")
    print(f"  training_curve.png")
    print(f"  reconstructions.png")
    print(f"  error_distribution.png")
    print(f"  clean_vs_poisoned.png")
    print(f"  epsilon_sensitivity.png")
    print(f"  detection_results.png")
    print(f"{sep}\n")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # ── Load everything ────────────────────
    print("Loading data...")
    x_train_clean = np.load("x_train_clean.npy")
    x_test_mixed  = np.load("x_test_mixed.npy")
    y_test_labels = np.load("y_test_labels.npy")

    # Separate clean and poisoned test images for comparison plots
    x_test_clean    = x_test_mixed[y_test_labels == 0]
    x_test_poisoned = x_test_mixed[y_test_labels == 1]

    print(f"Clean test images:    {len(x_test_clean)}")
    print(f"Poisoned test images: {len(x_test_poisoned)}")

    # ── Load detector ──────────────────────
    detector = PoisonDetector("saved_model/autoencoder_best.keras")
    detector.load_threshold("saved_model/threshold.npy")

    # ── Run all tests ──────────────────────

    # 1. Error distribution
    clean_errors, poisoned_errors = plot_error_distribution(
        detector,
        x_test_clean[:2000],
        x_test_poisoned[:2000]
    )

    # 2. Side by side comparison
    plot_clean_vs_poisoned_pairs(detector, x_test_clean, x_test_poisoned, n=5)

    # 3. Full evaluation metrics
    metrics = detector.evaluate(x_test_mixed, y_test_labels)

    # 4. Epsilon sensitivity (loads classifier from saved data)
    #    Comment this out if you don't want to retrain classifier
    try:
        from data_loader import load_data, build_classifier, train_classifier
        _, y_train, x_test_raw, y_test_raw = load_data()
        clf = build_classifier()
        clf = train_classifier(clf, np.load("x_train_clean.npy"), y_train,
                               x_test_raw, y_test_raw)
        test_epsilon_sensitivity(detector, x_test_raw[:500],
                                 y_test_raw[:500], clf)
    except Exception as e:
        print(f"Skipping epsilon test: {e}")

    # 5. Final report
    print_final_report(metrics, clean_errors, poisoned_errors, detector)

    print("All tests complete!")
