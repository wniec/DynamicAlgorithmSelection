from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    load_action_sequences,
)
from dynamicalgorithmselection.analysis.utils import (
    FUNCTION_TO_GROUP,
    GROUP_ORDER,
    REPO_ROOT,
)


def compute_transition_matrix(
    sequences: list[ActionSequence],
    algorithms: list[str],
    mode: str,
) -> dict[str, np.ndarray]:
    """Compute transition count matrices (prev→curr) aggregated by mode.

    Args:
        sequences: Loaded action sequences.
        algorithms: Ordered algorithm names.
        mode: One of ``"group"`` or ``"overall"``.

    Returns:
        Dict mapping label -> integer matrix of shape (n_algorithms, n_algorithms),
        where ``matrix[i, j]`` is the number of times algorithm ``i`` was followed
        by algorithm ``j``.
    """
    alg_index = {alg: k for k, alg in enumerate(algorithms)}
    n = len(algorithms)
    matrices: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((n, n), dtype=int))

    for seq in sequences:
        label = FUNCTION_TO_GROUP.get(seq.function_id, "Unknown") if mode == "group" else "Overall"
        for prev, curr in zip(seq.actions[:-1], seq.actions[1:]):
            if prev in alg_index and curr in alg_index:
                matrices[label][alg_index[prev], alg_index[curr]] += 1

    sorted_labels = _sort_groups(list(matrices)) if mode == "group" else list(matrices)
    return {label: matrices[label] for label in sorted_labels}


def plot_transition_counts(
    matrices: dict[str, np.ndarray],
    algorithms: list[str],
    output_path: Path,
) -> None:
    """Save a grid of transition-count heatmaps, one panel per label."""
    n = len(matrices)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 3.5),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    im = None
    for ax, (label, matrix) in zip(axes_flat, matrices.items()):
        vals = matrix.astype(float)
        vals[vals == 0] = np.nan
        im = ax.imshow(vals, aspect="auto", cmap="Blues")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms, fontsize=8)
        ax.set_xlabel("Algorithm (current step)", fontsize=8)
        ax.set_ylabel("Algorithm (previous step)", fontsize=8)
        for i in range(len(algorithms)):
            for j in range(len(algorithms)):
                cnt = matrix[i, j]
                if cnt > 0:
                    ax.text(j, i, str(cnt), ha="center", va="center", fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.8, label="Transition count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_empirical_probs(sequence: ActionSequence, algorithms: list[str]) -> np.ndarray:
    """One-hot encode actions into empirical per-checkpoint probabilities.

    Returns shape (n_algorithms, n_checkpoints).
    """
    alg_index = {alg: i for i, alg in enumerate(algorithms)}
    probs = np.zeros((len(algorithms), len(sequence.actions)))
    for t, action in enumerate(sequence.actions):
        probs[alg_index[action], t] = 1.0
    return probs


def _sort_groups(labels: list[str]) -> list[str]:
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(labels, key=lambda g: rank.get(g, math.inf))


def aggregate(
    sequences: list[ActionSequence],
    algorithms: list[str],
    mode: str,
    source: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Aggregate mean probabilities per label.

    Args:
        sequences: Loaded action sequences.
        algorithms: Ordered algorithm names (must match probabilities array order).
        mode: One of "function", "group", "overall".
        source: One of "actions" (empirical) or "probabilities" (policy).

    Returns:
        Tuple of (data, sample_sizes) where data maps label -> mean array
        of shape (n_algorithms, n_checkpoints).
    """
    groups: dict[str, list[np.ndarray]] = defaultdict(list)

    for seq in sequences:
        if mode == "function":
            label = seq.function_id
        elif mode == "group":
            label = FUNCTION_TO_GROUP.get(seq.function_id, "Unknown")
        else:
            label = "Overall"

        probs = (
            to_empirical_probs(seq, algorithms)
            if source == "actions"
            else np.array(seq.probabilities).T
        )
        groups[label].append(probs)

    if mode == "function":
        sorted_labels = sorted(groups, key=lambda f: int(f[1:]))
    elif mode == "group":
        sorted_labels = _sort_groups(list(groups))
    else:
        sorted_labels = list(groups)

    data = {label: np.mean(groups[label], axis=0) for label in sorted_labels}
    sample_sizes = {label: len(groups[label]) for label in sorted_labels}
    return data, sample_sizes


def plot_heatmap_grid(
    data: dict[str, np.ndarray],
    algorithms: list[str],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Save a publication-quality grid of heatmaps."""
    n = len(data)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 2.8),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    im = None
    for ax, (label, mean_probs) in zip(axes_flat, data.items()):
        im = ax.imshow(mean_probs, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=9)
        ax.set_yticks(range(len(algorithms)), algorithms, fontsize=8)
        ax.set_xticks(
            range(mean_probs.shape[1]),
            range(1, mean_probs.shape[1] + 1),
            fontsize=8,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Checkpoint", fontsize=8)
        ax.set_ylabel("Algorithm", fontsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.8, label="Selection probability")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    jsonl_path: Path,
    algorithms: list[str],
    output_dir: Path,
) -> None:
    """Generate all 6 heatmap plots for a given JSONL file."""
    sequences = load_action_sequences(jsonl_path)

    configs = [
        (
            "function",
            "actions",
            "function_action_probabilities.png",
        ),
        (
            "function",
            "probabilities",
            "function_policy_probabilities.png",
        ),
        (
            "group",
            "actions",
            "group_action_probabilities.png",
        ),
        (
            "group",
            "probabilities",
            "group_policy_probabilities.png",
        ),
        (
            "overall",
            "actions",
            "overall_action_probabilities.png",
        ),
        (
            "overall",
            "probabilities",
            "overall_policy_probabilities.png",
        ),
    ]

    for mode, source, filename in configs:
        data, _ = aggregate(sequences, algorithms, mode, source)
        output_path = output_dir / filename
        max_cols = 3 if mode == "group" else 4
        plot_heatmap_grid(data, algorithms, output_path, max_cols=max_cols)
        print(f"Saved: {output_path}")

    for mode, filename in [
        ("group", "group_transition_counts.png"),
        ("overall", "overall_transition_counts.png"),
    ]:
        matrices = compute_transition_matrix(sequences, algorithms, mode)
        output_path = output_dir / filename
        plot_transition_counts(matrices, algorithms, output_path)
        print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise DAS algorithm selection behaviour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        required=True,
        help="Algorithm names in probabilities array order (matches filename order).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: analysis_plots/<input stem>).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or REPO_ROOT / "analysis_plots" / args.input.stem
    generate_plots(args.input, args.algorithms, output_dir)


if __name__ == "__main__":
    main()
