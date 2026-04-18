"""Quantify policy concentration via KL divergence from uniform across dimensions.

For each checkpoint t, KL(p_t || uniform) = log(n) - H(p_t) where n is the
portfolio size and H is Shannon entropy.  A value of 0 means the policy is
perfectly uniform; log(n) means it is fully deterministic.

Usage::

    python -m dynamicalgorithmselection.analysis.policy_concentration \\
        --behaviour-dir /path/to/behaviour [--exp-types LOIO] [--cdb CDB1.0]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    BehaviourFile,
    discover_behaviour_files,
    load_action_sequences,
)
from dynamicalgorithmselection.analysis.utils import REPO_ROOT

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

_DIM_COLORS = {
    2: "#e41a1c",
    3: "#ff7f00",
    5: "#4daf4a",
    10: "#377eb8",
    20: "#984ea3",
    40: "#a65628",
}
_DIM_LINESTYLES = {
    2: "-",
    3: "--",
    5: "-.",
    10: ":",
    20: (0, (3, 1, 1, 1)),
    40: (0, (5, 1)),
}


def kl_from_uniform(probs: np.ndarray) -> np.ndarray:
    """KL(p_t || uniform) for each checkpoint.

    Args:
        probs: shape (n_algorithms, n_checkpoints), columns must sum to 1.

    Returns:
        shape (n_checkpoints,) in nats; range [0, log(n_algorithms)].
    """
    n_alg = probs.shape[0]
    p = np.clip(probs, 1e-12, 1.0)
    p = p / p.sum(axis=0, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0)
    return np.log(n_alg) - entropy


def _kl_for_sequence(seq: ActionSequence) -> np.ndarray:
    # seq.probabilities is (n_checkpoints, n_algorithms); transpose to (n_alg, n_ckpt)
    return kl_from_uniform(np.array(seq.probabilities).T)


def aggregate_kl(
    sequences: list[ActionSequence],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of per-checkpoint KL over all problem instances.

    Returns:
        (mean_kl, std_kl), each of shape (n_checkpoints,).
    """
    per_seq = np.stack([_kl_for_sequence(s) for s in sequences])
    return per_seq.mean(axis=0), per_seq.std(axis=0, ddof=0)


def compute_kl_by_dimension(
    behaviour_dir: Path,
    portfolio: str = "G3PCX_LMCMAES_SPSO",
    exp_types: tuple[str, ...] = ("LOIO",),
    cdb_filter: str | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load behaviour files and compute mean/std KL per dimension.

    All seeds and CV folds belonging to the same dimension (and optional CDB)
    are pooled before aggregation.

    Returns:
        Mapping dim -> (mean_kl, std_kl), each shape (n_checkpoints,).
    """
    all_files = discover_behaviour_files(
        behaviour_dir, portfolio=portfolio, exp_types=exp_types
    )
    if not all_files:
        raise FileNotFoundError(
            f"No behaviour files in {behaviour_dir} for portfolio={portfolio!r}."
        )

    dim_sequences: dict[int, list[ActionSequence]] = defaultdict(list)
    for bf in all_files:
        if cdb_filter and bf.cdb != cdb_filter:
            continue
        dim_sequences[bf.dim].extend(load_action_sequences(bf.path))

    if not dim_sequences:
        raise ValueError(
            f"No sequences matched (exp_types={exp_types}, cdb_filter={cdb_filter})."
        )

    return {dim: aggregate_kl(seqs) for dim, seqs in sorted(dim_sequences.items())}


def kl_summary_table(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    n_algorithms: int,
) -> pd.DataFrame:
    """Summary DataFrame with one row per dimension.

    Columns: Mean KL, Peak KL, Final KL, Normalised KL (divided by log(n_alg)).
    """
    max_kl = np.log(n_algorithms)
    rows = []
    for dim, (mean_kl, _) in sorted(results.items()):
        mean = float(mean_kl.mean())
        rows.append(
            {
                "Dimension": dim,
                "Mean KL": mean,
                "Peak KL": float(mean_kl.max()),
                "Final KL": float(mean_kl[-1]),
                "Norm. KL": mean / max_kl,
            }
        )
    df = pd.DataFrame(rows).set_index("Dimension")
    return df


def plot_kl_by_dimension(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    exp_type: str = "",
    cdb: str = "",
) -> None:
    """Line plot of mean KL divergence vs checkpoint, one line per dimension."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

        for dim, (mean_kl, std_kl) in sorted(results.items()):
            xs = np.arange(1, len(mean_kl) + 1)
            color = _DIM_COLORS.get(dim, f"C{dim}")
            ls = _DIM_LINESTYLES.get(dim, "-")
            ax.plot(
                xs,
                mean_kl,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                label=f"$d={dim}$",
            )
            ax.fill_between(
                xs,
                mean_kl - std_kl,
                mean_kl + std_kl,
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("KL divergence from uniform (nats)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(1, max(len(m) for m, _ in results.values()))

        title_parts = [p for p in [exp_type, cdb] if p]
        if title_parts:
            ax.set_title(" — ".join(title_parts), fontsize=9)

        ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy concentration analysis via KL divergence from uniform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("behaviour_dir", type=Path)
    parser.add_argument("--portfolio", default="G3PCX_LMCMAES_SPSO")
    parser.add_argument(
        "--algorithms",
        default=None,
        help=(
            "Comma-separated algorithm names in portfolio order. "
            "Defaults to splitting --portfolio on '_'."
        ),
    )
    parser.add_argument(
        "--exp-types",
        default="LOIO",
        help="Comma-separated experiment types, e.g. LOIO or LOIO,LOPO.",
    )
    parser.add_argument("--cdb", default=None, help="Filter to one CDB, e.g. CDB1.0.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    exp_types = tuple(t.strip() for t in args.exp_types.split(","))
    algorithms = (
        args.algorithms.split(",") if args.algorithms else args.portfolio.split("_")
    )
    n_algorithms = len(algorithms)
    output_dir = args.output_dir or (REPO_ROOT / "analysis_plots" / "concentration")

    results = compute_kl_by_dimension(
        args.behaviour_dir,
        portfolio=args.portfolio,
        exp_types=exp_types,
        cdb_filter=args.cdb,
    )
    print(f"Dimensions loaded: {sorted(results)}")

    label = "_".join(exp_types) + (f"_{args.cdb}" if args.cdb else "")
    plot_path = output_dir / f"kl_by_dimension_{label}.png"
    plot_kl_by_dimension(
        results,
        plot_path,
        exp_type=" / ".join(exp_types),
        cdb=args.cdb or "",
    )

    df = kl_summary_table(results, n_algorithms)
    print(f"\nKL divergence summary (log({n_algorithms}) = {np.log(n_algorithms):.4f} nats):")
    print(df.round(4).to_string())
    print("\nLaTeX:")
    print(df.round(4).to_latex())


if __name__ == "__main__":
    main()