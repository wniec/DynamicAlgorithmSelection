from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    BehaviourFile,
    discover_behaviour_files,
    load_action_sequences,
)
from dynamicalgorithmselection.analysis.utils import (
    FUNCTION_TO_GROUP,
    GROUP_ORDER,
    REPO_ROOT,
)

#: Per-algorithm colors (colorblind-friendly, readable in B&W via linestyles).
ALGO_COLORS: dict[str, str] = {
    "G3PCX": "#1f77b4",
    "LMCMAES": "#d62728",
    "SPSO": "#2ca02c",
}

ALGO_LINESTYLES: dict[str, str] = {
    "G3PCX": "-",
    "LMCMAES": "--",
    "SPSO": ":",
}

_RC: dict[str, object] = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

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


def _make_legend_handles(algorithms: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0], [0],
            color=ALGO_COLORS.get(alg, "black"),
            linestyle=ALGO_LINESTYLES.get(alg, "-"),
            linewidth=1.5,
            label=alg,
        )
        for alg in algorithms
    ]


def plot_lineplots_with_variability(
    mean_data: dict[str, np.ndarray],
    std_data: dict[str, np.ndarray],
    algorithms: list[str],
    output_path: Path,
    title: str = "",
    n_seeds: int | None = None,
) -> None:
    """Save a row of line plots (one per group) with ±1 std shading across seeds."""
    with plt.rc_context(_RC):
        n = len(mean_data)
        fig, axes = plt.subplots(
            1, n,
            figsize=(n * 2.8, 2.6),
            constrained_layout=True,
            sharey=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for ax, (label, mean_probs) in zip(axes_flat, mean_data.items()):
            std_probs = std_data.get(label, np.zeros_like(mean_probs))
            n_checkpoints = mean_probs.shape[1]
            xs = np.arange(1, n_checkpoints + 1)

            for i, alg in enumerate(algorithms):
                color = ALGO_COLORS.get(alg, f"C{i}")
                ls = ALGO_LINESTYLES.get(alg, "-")
                mu = mean_probs[i]
                sigma = std_probs[i]
                ax.plot(xs, mu, color=color, linestyle=ls, linewidth=1.5)
                ax.fill_between(xs, mu - sigma, mu + sigma, color=color, alpha=0.18)

            ax.set_title(label, fontsize=8, pad=3)
            ax.set_xlim(1, n_checkpoints)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(xs if n_checkpoints <= 10 else xs[::2])
            ax.set_xlabel("Checkpoint", fontsize=7)
            if ax is axes_flat[0]:
                seed_label = f"(n={n_seeds} seeds)" if n_seeds else ""
                ax.set_ylabel(f"Selection probability {seed_label}", fontsize=7)

        handles = _make_legend_handles(algorithms)
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(algorithms),
            frameon=False,
            fontsize=7,
        )

        if title:
            fig.suptitle(title, fontsize=9, y=1.02)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_heatmap_mean(
    mean_data: dict[str, np.ndarray],
    algorithms: list[str],
    output_path: Path,
    title: str = "",
) -> None:
    """Save a heatmap figure of mean selection probabilities per group."""
    n = len(mean_data)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, n,
            figsize=(n * 2.8, 2.2),
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for ax, (label, mean_probs) in zip(axes_flat, mean_data.items()):
            n_checkpoints = mean_probs.shape[1]
            xlabels = list(range(1, n_checkpoints + 1))
            im = ax.imshow(mean_probs, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(label, fontsize=8)
            ax.set_yticks(range(len(algorithms)), algorithms, fontsize=7)
            ax.set_xticks(range(n_checkpoints), xlabels, fontsize=7, rotation=45, ha="right")
            ax.set_xlabel("Checkpoint", fontsize=7)

        fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.8, label="Selection probability")

        if title:
            fig.suptitle(title, fontsize=9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_summary_lineplots(
    results: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    algorithms: list[str],
    output_path: Path,
    exp_type: str,
    source: str,
    n_seeds: int | None = None,
) -> None:
    """Save a summary grid: rows = dimensionalities, columns = function groups."""
    dims = sorted(results.keys())
    sample_mean = next(iter(next(iter(results.values()))[0].values()))
    n_checkpoints = sample_mean.shape[1]
    xs = np.arange(1, n_checkpoints + 1)

    groups = list(next(iter(results.values()))[0].keys())
    n_rows = len(dims)
    n_cols = len(groups)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 2.6, n_rows * 2.2),
            constrained_layout=True,
            sharey=True,
            sharex=True,
        )
        axes = np.atleast_2d(axes)

        for row_idx, dim in enumerate(dims):
            mean_data, std_data = results[dim]
            for col_idx, group in enumerate(groups):
                ax = axes[row_idx, col_idx]
                if group not in mean_data:
                    ax.set_visible(False)
                    continue

                mean_probs = mean_data[group]
                std_probs = std_data.get(group, np.zeros_like(mean_probs))

                for i, alg in enumerate(algorithms):
                    color = ALGO_COLORS.get(alg, f"C{i}")
                    ls = ALGO_LINESTYLES.get(alg, "-")
                    mu = mean_probs[i]
                    sigma = std_probs[i]
                    ax.plot(xs, mu, color=color, linestyle=ls, linewidth=1.2)
                    ax.fill_between(xs, mu - sigma, mu + sigma, color=color, alpha=0.18)

                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(1, n_checkpoints)

                if row_idx == 0:
                    ax.set_title(group, fontsize=8, pad=3)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Checkpoint", fontsize=7)
                    ax.set_xticks(xs if n_checkpoints <= 10 else xs[::2])
                if col_idx == 0:
                    prob_label = "Policy prob." if source == "probabilities" else "Action freq."
                    seed_label = f"(n={n_seeds})" if n_seeds else ""
                    ax.set_ylabel(f"$d={dim}$\n{prob_label} {seed_label}", fontsize=7)

        handles = _make_legend_handles(algorithms)
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(algorithms),
            frameon=False,
            fontsize=7,
        )

        source_label = "Policy probabilities" if source == "probabilities" else "Action frequencies"
        fig.suptitle(f"{exp_type} — {source_label}", fontsize=10, y=1.01)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _aggregate_per_seed(
    files_by_seed: dict[int, list[BehaviourFile]],
    algorithms: list[str],
    mode: str,
    source: str,
) -> dict[int, dict[str, np.ndarray]]:
    """For each seed, merge sequences from all CV folds and aggregate by group."""
    result: dict[int, dict[str, np.ndarray]] = {}
    for seed, bf_list in files_by_seed.items():
        all_sequences: list[ActionSequence] = []
        for bf in bf_list:
            all_sequences.extend(load_action_sequences(bf.path))
        data, _ = aggregate(all_sequences, algorithms, mode, source)
        result[seed] = data
    return result


def _compute_seed_statistics(
    per_seed_data: dict[int, dict[str, np.ndarray]],
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute mean and std across seeds for each label."""
    sample = next(iter(per_seed_data.values()))
    labels = list(sample.keys())
    if mode == "group":
        labels = _sort_groups(labels)

    mean_data: dict[str, np.ndarray] = {}
    std_data: dict[str, np.ndarray] = {}
    for label in labels:
        arrays = [data[label] for data in per_seed_data.values() if label in data]
        if not arrays:
            continue
        stacked = np.stack(arrays, axis=0)  # (n_seeds, n_alg, n_checkpoints)
        mean_data[label] = stacked.mean(axis=0)
        std_data[label] = stacked.std(axis=0, ddof=0)

    return mean_data, std_data


def generate_multi_experiment_plots(
    behaviour_dir: Path,
    algorithms: list[str],
    output_dir: Path,
    portfolio: str = "G3PCX_LMCMAES_SPSO",
    exp_types: tuple[str, ...] = ("LOIO", "LOPO"),
) -> None:
    """Discover all behaviour files and generate publication-quality figures.

    For each experiment type (LOIO/LOPO), dimensionality (2/3/5/10), and CDB value:

    * Loads all CV-fold files (3 seeds x matching CDB blocks).
    * Aggregates per seed across all CV folds.
    * Computes mean and std across seeds.
    * Saves line plots (mean +- std shading) and heatmap pairs (mean + std).

    Additionally saves summary figures per (exp_type, CDB) with all
    dimensionalities in rows and all function groups in columns.
    """
    all_files = discover_behaviour_files(behaviour_dir, portfolio=portfolio, exp_types=exp_types)
    if not all_files:
        raise FileNotFoundError(
            f"No behaviour files matching portfolio={portfolio!r} found in {behaviour_dir}."
        )

    # Group by (exp_type, dim, cdb) then by seed
    grouped: dict[tuple[str, int, str], dict[int, list[BehaviourFile]]] = defaultdict(lambda: defaultdict(list))
    for bf in all_files:
        grouped[(bf.exp_type, bf.dim, bf.cdb)][bf.seed].append(bf)

    mode = "group"
    sources = [("probabilities", "policy"), ("actions", "actions")]

    # summary[exp_type][cdb][src_key][dim] = (mean_data, std_data)
    summary: dict[str, dict[str, dict[str, dict[int, tuple[dict, dict]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for (exp_type, dim, cdb), files_by_seed in sorted(grouped.items()):
        n_seeds = len(files_by_seed)
        plots_dir = output_dir / exp_type / f"dim{dim}" / cdb.lower() / "plots"
        print(f"\n[{exp_type}  dim={dim}  {cdb}]  seeds={sorted(files_by_seed)}  folds/seed={len(next(iter(files_by_seed.values())))}")

        for source, src_key in sources:
            per_seed = _aggregate_per_seed(files_by_seed, algorithms, mode, source)
            mean_data, std_data = _compute_seed_statistics(per_seed, mode)

            summary[exp_type][cdb][src_key][dim] = (mean_data, std_data)

            lp_path = plots_dir / f"{src_key}_lineplot.png"
            plot_lineplots_with_variability(
                mean_data, std_data, algorithms, lp_path,
                title=f"{exp_type}  d={dim}  {cdb}  —  {'Policy probabilities' if src_key == 'policy' else 'Action frequencies'}",
                n_seeds=n_seeds,
            )
            print(f"  Saved: {lp_path}")

            hm_path = plots_dir / f"{src_key}_heatmap.png"
            plot_heatmap_mean(
                mean_data, algorithms, hm_path,
                title=f"{exp_type}  d={dim}  {cdb}  —  {'Policy probabilities' if src_key == 'policy' else 'Action frequencies'}",
            )
            print(f"  Saved: {hm_path}")

    # Summary figures per (exp_type, cdb): rows = dims, cols = groups
    for exp_type, cdb_dict in summary.items():
        for cdb, src_dict in cdb_dict.items():
            for source, src_key in sources:
                dim_results = src_dict[src_key]
                if not dim_results:
                    continue
                any_dim = next(iter(dim_results))
                n_seeds = len(grouped[(exp_type, any_dim, cdb)])
                summary_path = output_dir / exp_type / cdb.lower() / f"{src_key}_summary.png"
                plot_summary_lineplots(
                    dim_results, algorithms, summary_path,
                    exp_type=f"{exp_type} {cdb}",
                    source=source,
                    n_seeds=n_seeds,
                )
                print(f"\nSaved summary: {summary_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise DAS algorithm selection behaviour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "behaviour_dir",
        type=Path,
        help="Directory with behaviour JSONL files.",
    )
    parser.add_argument(
        "--algorithms",
        default="G3PCX,LMCMAES,SPSO",
        help=(
            "Comma-separated ordered list of algorithm names, e.g. "
            "MADDE,JDE21,NL_SHADE_RSP.  Required when any algorithm name "
            "contains an underscore.  Defaults to splitting --portfolio on '_', "
            "which is only correct when no algorithm name contains underscores."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or REPO_ROOT / "analysis_plots" / "multi_experiment"
    algorithms = [alg for alg in args.portfolio.split("_") if alg]
    generate_multi_experiment_plots(
        behaviour_dir=args.behaviour_dir,
        algorithms=algorithms,
        output_dir=output_dir,
        portfolio=args.portfolio,
    )


if __name__ == "__main__":
    main()
