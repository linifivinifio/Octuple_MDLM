import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _load_npy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def iter_sequences(arr: np.ndarray) -> Iterable[np.ndarray]:
    """Yield sequences shaped (T, C).

    Supports:
      - Ragged object arrays: arr.shape == (N,) where each item is (T_i, C)
      - Fixed arrays: arr.shape == (N, T, C)
      - Occasionally object dtype even for fixed arrays
    """
    if arr.ndim == 1:
        for item in arr:
            if item is None:
                continue
            seq = np.asarray(item)
            if seq.ndim != 2:
                raise ValueError(f"Expected 2D sequence (T,C), got shape {seq.shape}")
            yield seq
        return

    if arr.ndim == 3:
        for i in range(arr.shape[0]):
            seq = np.asarray(arr[i])
            if seq.ndim != 2:
                raise ValueError(f"Expected 2D sequence (T,C), got shape {seq.shape}")
            yield seq
        return

    raise ValueError(f"Unsupported array shape {arr.shape} (ndim={arr.ndim})")


def normalize_seq(seq: np.ndarray) -> np.ndarray:
    """Convert seq to a numeric 2D array if possible."""
    if seq.dtype == object:
        # Try int first (common for token IDs), then float.
        try:
            return seq.astype(np.int64)
        except Exception:
            return seq.astype(np.float64)
    return seq


def first_pass_min_max(seqs: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    min_per_c = None
    max_per_c = None

    for seq in seqs:
        seq = normalize_seq(seq)
        if seq.size == 0:
            continue

        if min_per_c is None:
            min_per_c = np.full((seq.shape[1],), np.inf, dtype=np.float64)
            max_per_c = np.full((seq.shape[1],), -np.inf, dtype=np.float64)

        if seq.shape[1] != min_per_c.shape[0]:
            raise ValueError(
                f"Inconsistent channel count: saw {seq.shape[1]} but expected {min_per_c.shape[0]}"
            )

        min_per_c = np.minimum(min_per_c, np.min(seq, axis=0))
        max_per_c = np.maximum(max_per_c, np.max(seq, axis=0))

    if min_per_c is None or max_per_c is None:
        raise ValueError("No sequences found (empty dataset?)")

    return min_per_c, max_per_c


def second_pass_hist(
    seqs: Iterable[np.ndarray],
    min_per_c: np.ndarray,
    max_per_c: np.ndarray,
    bins: int,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute hist counts per channel.

    Returns:
      - counts_per_c: list of arrays
      - edges_per_c: list of bin edges (len = bins+1)
    """
    channel_count = int(min_per_c.shape[0])
    counts_per_c = [np.zeros((bins,), dtype=np.int64) for _ in range(channel_count)]
    edges_per_c = []

    for c in range(channel_count):
        lo = float(min_per_c[c])
        hi = float(max_per_c[c])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if lo == hi:
            # Expand a tiny bit so histogram has a range.
            lo -= 0.5
            hi += 0.5
        edges_per_c.append(np.linspace(lo, hi, bins + 1, dtype=np.float64))

    for seq in seqs:
        seq = normalize_seq(seq)
        if seq.size == 0:
            continue

        for c in range(channel_count):
            values = seq[:, c]
            # Use numpy histogram for robustness (works for ints + floats)
            h, _ = np.histogram(values, bins=edges_per_c[c])
            counts_per_c[c] += h.astype(np.int64)

    return counts_per_c, edges_per_c


def _format_min_max(min_per_c: np.ndarray, max_per_c: np.ndarray) -> str:
    parts = []
    for c in range(min_per_c.shape[0]):
        parts.append(f"c{c}: [{min_per_c[c]:.0f}, {max_per_c[c]:.0f}]")
    return ", ".join(parts)


def plot_histograms(
    title: str,
    counts_per_c: list[np.ndarray],
    edges_per_c: list[np.ndarray],
    min_per_c: np.ndarray,
    max_per_c: np.ndarray,
    out_path: Path,
    ylog: bool,
) -> None:
    import matplotlib.pyplot as plt

    channel_count = len(counts_per_c)

    # Simple grid sizing.
    cols = 2 if channel_count > 1 else 1
    rows = (channel_count + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(title + "\n" + _format_min_max(min_per_c, max_per_c), fontsize=12)

    for c in range(channel_count):
        ax = axes[c // cols][c % cols]
        counts = counts_per_c[c]
        edges = edges_per_c[c]
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax.bar(centers, counts, width=(edges[1] - edges[0]), align="center")
        ax.set_title(f"Channel {c}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if ylog:
            ax.set_yscale("log")

    # Hide any unused axes
    for i in range(channel_count, rows * cols):
        axes[i // cols][i % cols].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_file(path: str, out_dir: str, bins: int, ylog: bool) -> None:
    arr = _load_npy(path)

    # First pass: min/max
    min_per_c, max_per_c = first_pass_min_max(iter_sequences(arr))

    # Second pass: hist
    counts_per_c, edges_per_c = second_pass_hist(iter_sequences(arr), min_per_c, max_per_c, bins=bins)

    # Print stats
    print(f"\n--- {path} ---")
    print(f"channels: {min_per_c.shape[0]}")
    for c in range(min_per_c.shape[0]):
        print(f"  c{c}: min={min_per_c[c]:.0f} max={max_per_c[c]:.0f}")

    # Plot
    out_path = Path(out_dir) / (Path(path).stem + "_hist.png")
    plot_histograms(
        title=f"Histograms: {path}",
        counts_per_c=counts_per_c,
        edges_per_c=edges_per_c,
        min_per_c=min_per_c,
        max_per_c=max_per_c,
        out_path=out_path,
        ylog=ylog,
    )
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel min/max and plot histograms for POP909 npy datasets."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "data/POP909_trio_octuple.npy",
            "data/POP909_trio.npy",
            "data/POP909_melody.npy",
            "data/POP909_melody_octuple.npy",
        ],
        help="List of .npy files to analyze",
    )
    parser.add_argument("--out_dir", default="plots/dataset_stats", help="Directory to save PNGs")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bin count per channel")
    parser.add_argument(
        "--ylog",
        action="store_true",
        help="Use log scale on histogram y-axis (useful for heavy-tailed counts)",
    )
    args = parser.parse_args()

    for f in args.files:
        if not os.path.exists(f):
            print(f"Skipping {f} (Not found)")
            continue
        analyze_file(f, out_dir=args.out_dir, bins=args.bins, ylog=args.ylog)


if __name__ == "__main__":
    main()
