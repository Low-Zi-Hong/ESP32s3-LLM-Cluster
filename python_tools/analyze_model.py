#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
from torch import tensor


# ============================================================
# Configuration
# ============================================================

PERCENTILES = [
    0.01,
    0.1,
    0.5,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9,
    99.99,
    99.999,
]

# Number of histogram bins
HIST_BINS = 400


# ============================================================
# Helpers
# ============================================================

def print_percentiles(values):
    print("\nPercentiles:")
    print("-" * 45)

    for p in PERCENTILES:
        value = np.percentile(values, p)
        print(f"{p:8.3f}% : {value:+.8e}")

    print("-" * 45)


def print_pruning_thresholds(values):
    abs_values = np.abs(values)

    print("\nPruning threshold analysis")
    print("=" * 70)

    thresholds = [
        1e-5,
        2.5e-5,
        5e-5,
        1e-4,
        2.5e-4,
        5e-4,
        1e-3,
        2.5e-3,
        5e-3,
        1e-2,
        2.5e-2,
        5e-2,
        0.1,
    ]

    print(f"{'Threshold':>15} {'Pruned':>15} {'Remaining':>15}")
    print("-" * 50)

    for threshold in thresholds:
        pruned = np.count_nonzero(abs_values < threshold)
        percentage = pruned / len(values) * 100

        print(
            f"{threshold:15.6g} "
            f"{percentage:14.4f}% "
            f"{100-percentage:14.4f}%"
        )


def analyze_tensor(name, tensor):
    # BF16 / FP16 / FP32 -> FP32
    flat = tensor.detach().float().cpu().numpy().reshape(-1)

    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "abs_mean": float(np.mean(np.abs(flat))),
        "abs_max": float(np.max(np.abs(flat))),
        "p01": float(np.percentile(flat, 0.1)),
        "p1": float(np.percentile(flat, 1)),
        "p99": float(np.percentile(flat, 99)),
        "p999": float(np.percentile(flat, 99.9)),
    }

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze weight distribution inside a safetensors model."
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to .safetensors file"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="weight_analysis",
        help="Output directory"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000_000,
        help="Maximum number of values used for plotting/statistics"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Safetensors Weight Distribution Analyzer")
    print("=" * 70)

    print(f"Model : {model_path}")
    print(f"Output: {out_dir}")

    # --------------------------------------------------------
    # Load tensors
    # --------------------------------------------------------

    all_values = []
    tensor_stats = []

    total_params = 0

    print("\nReading tensors...\n")

    with safe_open(
        model_path,
        framework="pt",
        device="cpu"
    ) as f:

        keys = list(f.keys())

        print(f"Tensor count: {len(keys)}")

        for i, key in enumerate(keys):

            tensor = f.get_tensor(key)

            # Only analyze floating point tensors
            if not tensor.is_floating_point():
                print(
                    f"[SKIP] {key} "
                    f"(dtype={tensor.dtype})"
                )
                continue

            flat = tensor.detach().float().cpu().numpy().reshape(-1)

            total_params += flat.size

            stats = analyze_tensor(key, tensor)
            tensor_stats.append(stats)

            # ------------------------------------------------
            # Collect samples
            # ------------------------------------------------

            remaining = args.max_samples - sum(
                len(x) for x in all_values
            )

            if remaining > 0:

                if len(flat) <= remaining:
                    all_values.append(flat)
                else:
                    # Random sample if tensor is huge
                    rng = np.random.default_rng(42)

                    idx = rng.choice(
                        len(flat),
                        size=remaining,
                        replace=False
                    )

                    all_values.append(flat[idx])

            print(
                f"[{i+1:4d}/{len(keys)}] "
                f"{key:60s} "
                f"{tuple(tensor.shape)}"
            )

    values = np.concatenate(all_values)

    print("\n")
    print("=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)

    print(f"Total parameters : {total_params:,}")
    print(f"Samples analyzed : {len(values):,}")

    print(f"\nMin              : {np.min(values):+.8e}")
    print(f"Max              : {np.max(values):+.8e}")
    print(f"Mean             : {np.mean(values):+.8e}")
    print(f"Std              : {np.std(values):+.8e}")

    print(f"\nAbs mean         : {np.mean(np.abs(values)):+.8e}")
    print(f"Abs max          : {np.max(np.abs(values)):+.8e}")

    print_percentiles(values)

    print_pruning_thresholds(values)

    # ========================================================
    # Histogram
    # ========================================================

    print("\nGenerating histograms...")

    # --------------------------------------------------------
    # Full distribution
    # --------------------------------------------------------

    plt.figure(figsize=(14, 8))

    plt.hist(
        values,
        bins=HIST_BINS,
        density=True
    )

    plt.title("Global Weight Distribution")
    plt.xlabel("Weight")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        out_dir / "weight_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Zoom around zero
    # --------------------------------------------------------

    p01 = np.percentile(values, 0.1)
    p999 = np.percentile(values, 99.9)

    plt.figure(figsize=(14, 8))

    plt.hist(
        values,
        bins=HIST_BINS,
        range=(p01, p999),
        density=True
    )

    plt.title(
        "Weight Distribution "
        "(0.1% - 99.9% percentile range)"
    )

    plt.xlabel("Weight")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        out_dir / "weight_distribution_zoom.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # Absolute weight distribution
    # ========================================================

    abs_values = np.abs(values)

    plt.figure(figsize=(14, 8))

    plt.hist(
        abs_values,
        bins=HIST_BINS,
        density=True
    )

    plt.title("|Weight| Distribution")
    plt.xlabel("|Weight|")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        out_dir / "absolute_weight_distribution.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # Log absolute distribution
    # ========================================================

    positive = abs_values[abs_values > 0]

    plt.figure(figsize=(14, 8))

    plt.hist(
        np.log10(positive),
        bins=HIST_BINS,
        density=True
    )

    plt.title("log10(|Weight|) Distribution")
    plt.xlabel("log10(|Weight|)")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        out_dir / "log_absolute_weight_distribution.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # Per-tensor statistics
    # ========================================================

    with open(
        out_dir / "tensor_statistics.json",
        "w"
    ) as f:

        json.dump(
            tensor_stats,
            f,
            indent=2
        )

    print("\nDone.")

    print("\nGenerated:")
    print("  weight_distribution.png")
    print("  weight_distribution_zoom.png")
    print("  absolute_weight_distribution.png")
    print("  log_absolute_weight_distribution.png")
    print("  tensor_statistics.json")


if __name__ == "__main__":
    main()