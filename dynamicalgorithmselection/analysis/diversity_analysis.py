from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    DiversitySequence,
    load_action_sequences,
    load_diversity_sequences,
)
from dynamicalgorithmselection.analysis.utils import (
    FUNCTION_TO_GROUP,
    GROUP_ORDER,
    REPO_ROOT,
)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def build_transition_records(
    action_sequences: list[ActionSequence],
    diversity_sequences: list[DiversitySequence],
) -> pd.DataFrame:
    """Build a flat DataFrame aligning actions with diversity values.

    For each problem instance and each step k (1 ≤ k ≤ T−1), one record is
    created with the following columns:

    - ``problem_key``, ``function_id``, ``instance_id``, ``dimension``,
      ``group`` — problem metadata.
    - ``step`` — 1-based index (1 = first decision after the start).
    - ``prev_action``, ``curr_action`` — algorithms at steps k−1 and k.
    - ``decision_type`` — ``"same"`` or ``"switch"``.
    - ``transition`` — ``"ALGO_A→ALGO_A"`` or ``"ALGO_A→ALGO_B"``.
    - ``diversity`` — diversity value at step k (d[k]).
    - ``diversity_change`` — d[k] − d[k−1]; NaN for k=1 because d[0] is the
      dropped artefact and cannot serve as a meaningful baseline.
    - ``diversity_increased`` — ``True`` / ``False``; NaN for k=1.
    """
    diversity_map = {seq.problem_key: seq for seq in diversity_sequences}
    records = []

    for action_seq in action_sequences:
        pk = action_seq.problem_key
        if pk not in diversity_map:
            continue
        div_seq = diversity_map[pk]

        T = len(action_seq.actions)
        if len(div_seq.diversity) != T - 1:
            raise ValueError(
                f"Length mismatch for {pk}: {T} actions but "
                f"{len(div_seq.diversity)} diversity values (expected {T - 1})."
            )

        group = FUNCTION_TO_GROUP.get(action_seq.function_id, "Unknown")

        for k in range(1, T):
            prev_action = action_seq.actions[k - 1]
            curr_action = action_seq.actions[k]
            diversity = div_seq.diversity[k - 1]

            if k >= 2:
                diversity_change = float(div_seq.diversity[k - 1] - div_seq.diversity[k - 2])
                diversity_increased = diversity_change > 0.0
            else:
                diversity_change = np.nan
                diversity_increased = np.nan

            records.append(
                {
                    "problem_key": pk,
                    "function_id": action_seq.function_id,
                    "instance_id": action_seq.instance_id,
                    "dimension": action_seq.dimension,
                    "group": group,
                    "step": k,
                    "prev_action": prev_action,
                    "curr_action": curr_action,
                    "decision_type": "same" if prev_action == curr_action else "switch",
                    "transition": f"{prev_action}→{curr_action}",
                    "diversity": diversity,
                    "diversity_change": diversity_change,
                    "diversity_increased": diversity_increased,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _violin_or_box(
    ax: plt.Axes,
    data_groups: list[tuple[str, np.ndarray]],
    ylabel: str,
    title: str,
    hline: float | None = None,
) -> None:
    """Draw a violin plot with mean markers for the given groups."""
    labels = [label for label, _ in data_groups]
    arrays = [arr for _, arr in data_groups]

    parts = ax.violinplot(arrays, positions=range(1, len(labels) + 1), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)

    means = [arr.mean() for arr in arrays]
    ax.scatter(
        range(1, len(labels) + 1),
        means,
        marker="^",
        color="crimson",
        zorder=5,
        s=60,
        label="Mean",
    )

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _hbar(
    ax: plt.Axes,
    series: pd.Series,
    counts: pd.Series,
    color: str,
    xlabel: str,
    title: str,
) -> None:
    """Draw a horizontal bar chart with per-bar count annotations and a mean line."""
    algos = series.index.tolist()
    ax.barh(algos, series.values, color=color, alpha=0.8)
    mean_val = series.mean()
    ax.axvline(mean_val, color="gray", linestyle="--", linewidth=0.8, label=f"Mean ({mean_val:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    x_max = series.max() if series.max() > 0 else 1.0
    for i, (val, cnt) in enumerate(zip(series.values, counts.values)):
        ax.text(val + 0.01 * x_max, i, f"n={int(cnt)}", va="center", fontsize=8)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_diversity_distributions(records: pd.DataFrame, output_path: Path) -> Path:
    """Violin plots comparing diversity and diversity change for same vs switch."""
    same = records[records["decision_type"] == "same"]
    switch = records[records["decision_type"] == "switch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _violin_or_box(
        axes[0],
        [
            ("Same\nalgorithm", same["diversity"].dropna().to_numpy()),
            ("Switch\nalgorithm", switch["diversity"].dropna().to_numpy()),
        ],
        ylabel="Population diversity",
        title="Diversity level by decision type",
    )

    change_same = same["diversity_change"].dropna().to_numpy()
    change_switch = switch["diversity_change"].dropna().to_numpy()
    _violin_or_box(
        axes[1],
        [
            ("Same\nalgorithm", change_same),
            ("Switch\nalgorithm", change_switch),
        ],
        ylabel="Diversity change (Δ)",
        title="Diversity change by decision type",
        hline=0.0,
    )

    for pos, arr in [(1, change_same), (2, change_switch)]:
        pct = 100.0 * (arr > 0).mean() if len(arr) > 0 else 0.0
        ylim = axes[1].get_ylim()
        axes[1].text(
            pos,
            ylim[1] * 0.92 if ylim[1] != 0 else 0.1,
            f"{pct:.0f}% ↑",
            ha="center",
            fontsize=8,
            color="steelblue",
        )

    fig.suptitle("Population diversity: same-algorithm vs switch decisions", fontsize=13)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_transition_heatmap(records: pd.DataFrame, output_path: Path) -> Path:
    """Heatmap of mean diversity change and transition counts for each (from→to) pair."""
    change_records = records.dropna(subset=["diversity_change"])
    algorithms = sorted(
        set(records["prev_action"].unique()) | set(records["curr_action"].unique())
    )

    change_pivot = (
        change_records.groupby(["prev_action", "curr_action"])["diversity_change"]
        .mean()
        .unstack()
        .reindex(index=algorithms, columns=algorithms)
    )
    count_pivot = (
        change_records.groupby(["prev_action", "curr_action"])["diversity_change"]
        .count()
        .unstack()
        .reindex(index=algorithms, columns=algorithms)
        .fillna(0)
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax = np.nanmax(np.abs(change_pivot.values))
    im1 = axes[0].imshow(change_pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Mean diversity change (Δ) per transition")
    axes[0].set_xlabel("Algorithm selected (current step)")
    axes[0].set_ylabel("Algorithm selected (previous step)")
    axes[0].set_xticks(range(len(algorithms)))
    axes[0].set_xticklabels(algorithms, rotation=30, ha="right")
    axes[0].set_yticks(range(len(algorithms)))
    axes[0].set_yticklabels(algorithms)
    fig.colorbar(im1, ax=axes[0], label="Mean Δdiversity")

    for i, row_algo in enumerate(algorithms):
        for j, col_algo in enumerate(algorithms):
            val = change_pivot.loc[row_algo, col_algo] if row_algo in change_pivot.index and col_algo in change_pivot.columns else np.nan
            if not np.isnan(val):
                axes[0].text(
                    j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) > 0.4 * vmax else "black",
                )

    count_vals = count_pivot.values.astype(float)
    count_vals[count_vals == 0] = np.nan
    im2 = axes[1].imshow(count_vals, aspect="auto", cmap="Blues")
    axes[1].set_title("Transition counts")
    axes[1].set_xlabel("Algorithm selected (current step)")
    axes[1].set_ylabel("Algorithm selected (previous step)")
    axes[1].set_xticks(range(len(algorithms)))
    axes[1].set_xticklabels(algorithms, rotation=30, ha="right")
    axes[1].set_yticks(range(len(algorithms)))
    axes[1].set_yticklabels(algorithms)
    fig.colorbar(im2, ax=axes[1], label="Count")

    for i, row_algo in enumerate(algorithms):
        for j, col_algo in enumerate(algorithms):
            cnt = count_pivot.loc[row_algo, col_algo] if row_algo in count_pivot.index and col_algo in count_pivot.columns else 0
            if cnt > 0:
                axes[1].text(j, i, str(cnt), ha="center", va="center", fontsize=8)

    fig.suptitle("Diversity change by algorithm transition (from→to)", fontsize=13)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_increase_probability(records: pd.DataFrame, output_path: Path) -> Path:
    """Bar chart of P(diversity increases) for each decision type and transition."""
    change_records = records.dropna(subset=["diversity_change"])

    transition_stats = (
        change_records.groupby(["transition", "decision_type"])["diversity_increased"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "p_increase", "count": "n"})
    )
    transition_stats = transition_stats.sort_values(
        ["decision_type", "p_increase"], ascending=[True, False]
    )

    same_rows = transition_stats[transition_stats["decision_type"] == "same"]
    switch_rows = transition_stats[transition_stats["decision_type"] == "switch"]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharey=False, gridspec_kw={"width_ratios": [1, 3]}
    )

    for ax, rows, color, title in [
        (axes[0], same_rows, "steelblue", "Same-algorithm decisions"),
        (axes[1], switch_rows, "darkorange", "Switch decisions"),
    ]:
        bars = ax.barh(rows["transition"], rows["p_increase"], color=color, alpha=0.8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(diversity increases)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        for bar, row in zip(bars, rows.itertuples()):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={row.n}", va="center", fontsize=8,
            )

    fig.suptitle("Probability that diversity increases by decision type", fontsize=13)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_diversity_change_by_group(records: pd.DataFrame, output_path: Path) -> Path:
    """For each BBOB group, compare mean diversity change for same vs switch decisions."""
    change_records = records.dropna(subset=["diversity_change"])
    groups_present = [g for g in GROUP_ORDER if g in change_records["group"].unique()]

    fig, axes = plt.subplots(
        1, len(groups_present), figsize=(4 * len(groups_present), 5), sharey=True
    )
    if len(groups_present) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, groups_present):
        group_data = change_records[change_records["group"] == group_name]
        same_vals = group_data[group_data["decision_type"] == "same"]["diversity_change"].to_numpy()
        switch_vals = group_data[group_data["decision_type"] == "switch"]["diversity_change"].to_numpy()

        _violin_or_box(
            ax,
            [("Same", same_vals), ("Switch", switch_vals)],
            ylabel="Diversity change (Δ)" if ax is axes[0] else "",
            title=group_name,
            hline=0.0,
        )

    fig.suptitle("Diversity change (Δ) by BBOB group and decision type", fontsize=13)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_significant_diversity_changes(
    records: pd.DataFrame,
    output_path: Path,
    drop_percentile: float = 5.0,
    spike_percentile: float = 95.0,
) -> Path:
    """Show which algorithms are disproportionately linked to large diversity swings."""
    change_records = records.dropna(subset=["diversity_change"]).copy()

    drop_thresh = float(np.percentile(change_records["diversity_change"], drop_percentile))
    spike_thresh = float(np.percentile(change_records["diversity_change"], spike_percentile))

    sig_drops = change_records[change_records["diversity_change"] <= drop_thresh]
    sig_spikes = change_records[change_records["diversity_change"] >= spike_thresh]

    total_steps = change_records.groupby("curr_action")["step"].count().rename("total")
    drop_counts = sig_drops.groupby("curr_action")["step"].count().rename("drops")
    spike_counts = sig_spikes.groupby("curr_action")["step"].count().rename("spikes")

    algo_stats = pd.concat([total_steps, drop_counts, spike_counts], axis=1).fillna(0)
    algo_stats["drop_rate"] = algo_stats["drops"] / algo_stats["total"]
    algo_stats["spike_rate"] = algo_stats["spikes"] / algo_stats["total"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    drop_sorted = algo_stats.sort_values("drop_rate", ascending=False)
    _hbar(
        axes[0],
        drop_sorted["drop_rate"],
        drop_sorted["drops"],
        color="crimson",
        xlabel=(
            f"Fraction of steps with Δdiversity ≤ {drop_thresh:.4f}\n"
            f"(bottom {drop_percentile:.0f}% of all changes)"
        ),
        title="Significant drops per algorithm",
    )

    spike_sorted = algo_stats.sort_values("spike_rate", ascending=False)
    _hbar(
        axes[1],
        spike_sorted["spike_rate"],
        spike_sorted["spikes"],
        color="steelblue",
        xlabel=(
            f"Fraction of steps with Δdiversity ≥ {spike_thresh:.4f}\n"
            f"(top {100 - spike_percentile:.0f}% of all changes)"
        ),
        title="Significant spikes per algorithm",
    )

    fig.suptitle(
        f"Significant diversity changes by algorithm\n"
        f"drop ≤ {drop_thresh:.4f}  |  spike ≥ {spike_thresh:.4f}",
        fontsize=13,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_plots(
    actions_path: Path,
    diversity_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate all diversity analysis plots for a given actions/diversity pair."""
    records = build_transition_records(
        load_action_sequences(actions_path),
        load_diversity_sequences(diversity_path),
    )
    output_dir = Path(output_dir)

    return {
        "diversity_distributions": plot_diversity_distributions(
            records, output_dir / "diversity_distributions.png"
        ),
        "transition_heatmap": plot_transition_heatmap(
            records, output_dir / "transition_heatmap.png"
        ),
        "increase_probability": plot_increase_probability(
            records, output_dir / "increase_probability.png"
        ),
        "diversity_change_by_group": plot_diversity_change_by_group(
            records, output_dir / "diversity_change_by_group.png"
        ),
        "significant_changes": plot_significant_diversity_changes(
            records, output_dir / "significant_changes.png"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse how algorithm-switching decisions affect population diversity."
    )
    parser.add_argument("--input", type=Path, required=True, help="Actions JSONL file.")
    parser.add_argument("--diversity", type=Path, required=True, help="Diversity JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots.")
    args = parser.parse_args()

    output_dir = args.output_dir or REPO_ROOT / "analysis_plots" / args.input.stem / "diversity"
    outputs = generate_plots(args.input, args.diversity, output_dir)

    print(f"Saved diversity analysis report to: {output_dir}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
