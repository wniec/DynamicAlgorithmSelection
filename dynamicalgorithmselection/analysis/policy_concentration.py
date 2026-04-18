"""Quantify how much learned policies differ from the CDB=1.0 baseline.

For each (dimension, CDB, checkpoint) cell, compute KL(p_cdb || p_cdb1.0) where
both distributions are the mean policy averaged across all matching sequences.
Visualised as four heatmaps (one per dimension): x-axis = CDB, y-axis = checkpoint.

Usage::

    python -m dynamicalgorithmselection.analysis.policy_concentration \\
        --behaviour-dir /path/to/behaviour [--exp-types LOIO] [--ref-cdb CDB1.0]
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
}


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats with epsilon smoothing."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _mean_policy(sequences: list[ActionSequence]) -> np.ndarray:
    """Mean policy across sequences.

    seq.probabilities is (n_checkpoints, n_algorithms); returns same shape.
    """
    return np.mean([np.array(s.probabilities) for s in sequences], axis=0)


def compute_kl_heatmaps(
    behaviour_dir: Path,
    portfolio: str = "G3PCX_LMCMAES_SPSO",
    exp_types: tuple[str, ...] = ("LOIO",),
    ref_cdb: str = "CDB1.0",
) -> dict[int, tuple[np.ndarray, list[str]]]:
    """Compute per-checkpoint KL divergence from the CDB=1.0 reference policy.

    Args:
        behaviour_dir: Directory containing behaviour JSONL files.
        portfolio: Portfolio string used to filter filenames.
        exp_types: Experiment types to include (e.g. ``("LOIO",)``).
        ref_cdb: CDB label to use as the reference distribution.

    Returns:
        Mapping ``dim -> (heatmap, cdb_labels)`` where ``heatmap`` has shape
        ``(n_checkpoints, n_cdbs)`` and ``cdb_labels`` lists CDB strings in
        ascending order.
    """
    all_files = discover_behaviour_files(
        behaviour_dir, portfolio=portfolio, exp_types=exp_types
    )
    if not all_files:
        raise FileNotFoundError(
            f"No behaviour files in {behaviour_dir} for portfolio={portfolio!r}."
        )

    dim_cdb_seqs: dict[tuple[int, str], list[ActionSequence]] = defaultdict(list)
    for bf in all_files:
        dim_cdb_seqs[(bf.dim, bf.cdb)].extend(load_action_sequences(bf.path))

    dims = sorted({k[0] for k in dim_cdb_seqs})
    results: dict[int, tuple[np.ndarray, list[str]]] = {}

    for dim in dims:
        ref_key = (dim, ref_cdb)
        if ref_key not in dim_cdb_seqs:
            print(f"Warning: no {ref_cdb} sequences for dim={dim}, skipping.")
            continue

        ref_policy = _mean_policy(dim_cdb_seqs[ref_key])  # (n_ckpt, n_alg)
        n_checkpoints = ref_policy.shape[0]

        cdbs = sorted(
            {k[1] for k in dim_cdb_seqs if k[0] == dim},
            key=lambda c: float(c.replace("CDB", "")),
        )

        heatmap = np.zeros((n_checkpoints, len(cdbs)))
        for j, cdb in enumerate(cdbs):
            policy = _mean_policy(dim_cdb_seqs[(dim, cdb)])  # (n_ckpt, n_alg)
            for t in range(n_checkpoints):
                heatmap[t, j] = _kl(policy[t], ref_policy[t])

        results[dim] = (heatmap, cdbs)

    return results


def plot_kl_heatmaps(
    results: dict[int, tuple[np.ndarray, list[str]]],
    output_path: Path,
    ref_cdb: str = "CDB1.0",
    exp_type: str = "",
) -> None:
    """Four side-by-side heatmaps, one per dimension.

    X-axis: CDB value, Y-axis: checkpoint index.
    """
    dims = sorted(results)
    n = len(dims)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(n * 2.6, 3.6),
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        # Shared colour scale across all panels
        vmax = max(heatmap.max() for heatmap, _ in results.values())

        im = None
        for ax, dim in zip(axes_flat, dims):
            heatmap, cdbs = results[dim]
            n_checkpoints = heatmap.shape[0]

            im = ax.imshow(
                heatmap,
                aspect="auto",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=vmax,
                origin="upper",
            )

            cdb_labels = [c.replace("CDB", "") for c in cdbs]
            ax.set_xticks(range(len(cdbs)), cdb_labels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(
                range(n_checkpoints),
                [str(t + 1) for t in range(n_checkpoints)],
                fontsize=6,
            )
            ax.set_xlabel("CDB", fontsize=8)
            ax.set_title(f"$d={dim}$", fontsize=9)
            if ax is axes_flat[0]:
                ax.set_ylabel("Checkpoint", fontsize=8)

        if im is not None:
            fig.colorbar(
                im,
                ax=axes_flat.tolist(),
                shrink=0.8,
                label=f"KL divergence from {ref_cdb} (nats)",
            )

        title_parts = [p for p in [exp_type, f"reference: {ref_cdb}"] if p]
        if title_parts:
            fig.suptitle(" — ".join(title_parts), fontsize=9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def kl_summary_table(
    results: dict[int, tuple[np.ndarray, list[str]]],
) -> pd.DataFrame:
    """Summary DataFrame: mean KL per (dimension, CDB)."""
    rows = []
    for dim, (heatmap, cdbs) in sorted(results.items()):
        for j, cdb in enumerate(cdbs):
            rows.append(
                {
                    "Dimension": dim,
                    "CDB": cdb.replace("CDB", ""),
                    "Mean KL": float(heatmap[:, j].mean()),
                    "Peak KL": float(heatmap[:, j].max()),
                    "Final KL": float(heatmap[-1, j]),
                }
            )
    return pd.DataFrame(rows).set_index(["Dimension", "CDB"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy concentration: KL heatmaps vs CDB=1.0 reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("behaviour_dir", type=Path)
    parser.add_argument("--portfolio", default="G3PCX_LMCMAES_SPSO")
    parser.add_argument("--exp-types", default="LOIO")
    parser.add_argument("--ref-cdb", default="CDB1.0")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    exp_types = tuple(t.strip() for t in args.exp_types.split(","))
    output_dir = args.output_dir or (REPO_ROOT / "analysis_plots" / "concentration")

    results = compute_kl_heatmaps(
        args.behaviour_dir,
        portfolio=args.portfolio,
        exp_types=exp_types,
        ref_cdb=args.ref_cdb,
    )
    print(f"Dimensions loaded: {sorted(results)}")

    label = "_".join(exp_types)
    plot_path = output_dir / f"kl_heatmap_{label}.png"
    plot_kl_heatmaps(
        results,
        plot_path,
        ref_cdb=args.ref_cdb,
        exp_type=" / ".join(exp_types),
    )

    df = kl_summary_table(results)
    print("\nKL divergence summary (mean over checkpoints):")
    print(df.round(4).to_string())
    print("\nLaTeX:")
    print(df.round(4).to_latex())


if __name__ == "__main__":
    main()