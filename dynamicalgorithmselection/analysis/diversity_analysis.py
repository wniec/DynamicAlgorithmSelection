from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamicalgorithmselection.analysis.loading import ActionSequence, load_action_sequences
from dynamicalgorithmselection.analysis.utils import (
    FUNCTION_TO_GROUP,
    GROUP_ORDER,
    PROBLEM_KEY_RE,
    REPO_ROOT,
    sanitize_filename,
)

DEFAULT_ACTIONS_PATH = (
    REPO_ROOT / "DAS_CV_G3PCX_CMAES_MADDE_PG_CV-LOPO_CDB1.5_DIM10_SEED123.jsonl"
)
DEFAULT_DIVERSITY_PATH = (
    REPO_ROOT / "DAS_CV_G3PCX_CMAES_MADDE_PG_CV-LOPO_CDB1.5_DIM10_SEED123_diversity.jsonl"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiversitySequence:
    """One problem instance and its population diversity values across checkpoints.

    The first diversity value in the raw file (always 0.0) is an artefact and
    is skipped during loading, so ``diversity[0]`` corresponds to checkpoint 1.
    """

    problem_key: str
    function_id: str
    instance_id: str
    dimension: int
    diversity: tuple[float, ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_diversity_sequences(jsonl_path: str | Path) -> list[DiversitySequence]:
    """Load population diversity sequences from a JSONL file.

    Each line must contain a single ``{problem_key: [float, ...]}`` entry.
    The first value is always 0.0 (artefact) and is skipped automatically.

    Args:
        jsonl_path: Path to the diversity JSONL file.

    Returns:
        List of :class:`DiversitySequence` objects (first value already dropped).
    """
    path = Path(jsonl_path)
    sequences: list[DiversitySequence] = []

    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if len(payload) != 1:
                raise ValueError(
                    f"Expected exactly one entry on line {line_number}, got {len(payload)}."
                )

            problem_key, diversity_values = next(iter(payload.items()))
            match = PROBLEM_KEY_RE.fullmatch(problem_key)
            if match is None:
                raise ValueError(
                    f"Unsupported problem key on line {line_number}: {problem_key!r}"
                )
            if not isinstance(diversity_values, list) or len(diversity_values) < 2:
                raise ValueError(
                    f"Expected a list with at least 2 values for {problem_key!r}."
                )

            function_id, instance_id, dimension = match.groups()
            # Drop the first value (always 0.0, artefact).
            sequences.append(
                DiversitySequence(
                    problem_key=problem_key,
                    function_id=function_id,
                    instance_id=instance_id,
                    dimension=int(dimension),
                    diversity=tuple(float(v) for v in diversity_values[1:]),
                )
            )

    if not sequences:
        raise ValueError(f"No diversity sequences found in {path}.")

    return sequences


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

    Args:
        action_sequences: Loaded action sequences.
        diversity_sequences: Loaded diversity sequences (first value skipped).

    Returns:
        DataFrame with one row per (problem_instance, step).
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
            # div_seq.diversity[j] = d[j+1], so diversity at step k is index k-1.
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


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


@dataclass
class DiversityAnalyser:
    """Analyses the relationship between algorithm-switching decisions and diversity."""

    action_sequences: list[ActionSequence]
    diversity_sequences: list[DiversitySequence]

    @classmethod
    def from_jsonl_pair(
        cls,
        actions_path: str | Path,
        diversity_path: str | Path,
    ) -> "DiversityAnalyser":
        return cls(
            action_sequences=load_action_sequences(actions_path),
            diversity_sequences=load_diversity_sequences(diversity_path),
        )

    def build_records(self) -> pd.DataFrame:
        """Return the flat transition records DataFrame."""
        return build_transition_records(self.action_sequences, self.diversity_sequences)

    # ------------------------------------------------------------------
    # Plot 1 — diversity distributions: same vs switch
    # ------------------------------------------------------------------

    def plot_diversity_distributions(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
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

        # Annotate with fraction of positive changes
        for ax, arr_same, arr_switch in [
            (axes[1], change_same, change_switch)
        ]:
            for pos, arr, label in [
                (1, arr_same, "Same"),
                (2, arr_switch, "Switch"),
            ]:
                pct = 100.0 * (arr > 0).mean() if len(arr) > 0 else 0.0
                ax.text(
                    pos,
                    ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] != 0 else 0.1,
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

    # ------------------------------------------------------------------
    # Plot 2 — transition heatmaps: mean Δdiversity and counts
    # ------------------------------------------------------------------

    def plot_transition_heatmap(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
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

        # Mean diversity change
        vmax = np.nanmax(np.abs(change_pivot.values))
        im1 = axes[0].imshow(
            change_pivot.values,
            aspect="auto",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
        )
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

        # Transition counts
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

    # ------------------------------------------------------------------
    # Plot 3 — probability that diversity increases per transition
    # ------------------------------------------------------------------

    def plot_increase_probability(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Bar chart of P(diversity increases) for each decision type and transition."""
        change_records = records.dropna(subset=["diversity_change"])

        # Per-transition (from→to) statistics
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

        # Same-algorithm decisions
        ax_same = axes[0]
        bars_s = ax_same.barh(
            same_rows["transition"],
            same_rows["p_increase"],
            color="steelblue",
            alpha=0.8,
        )
        ax_same.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax_same.set_xlim(0, 1)
        ax_same.set_xlabel("P(diversity increases)")
        ax_same.set_title("Same-algorithm decisions")
        ax_same.grid(True, alpha=0.3, axis="x")
        for bar, row in zip(bars_s, same_rows.itertuples()):
            ax_same.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={row.n}", va="center", fontsize=8,
            )

        # Switch decisions
        ax_switch = axes[1]
        bars_w = ax_switch.barh(
            switch_rows["transition"],
            switch_rows["p_increase"],
            color="darkorange",
            alpha=0.8,
        )
        ax_switch.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax_switch.set_xlim(0, 1)
        ax_switch.set_xlabel("P(diversity increases)")
        ax_switch.set_title("Switch decisions")
        ax_switch.grid(True, alpha=0.3, axis="x")
        for bar, row in zip(bars_w, switch_rows.itertuples()):
            ax_switch.text(
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

    # ------------------------------------------------------------------
    # Plot 4 — per-BBOB-group breakdown
    # ------------------------------------------------------------------

    def plot_diversity_change_by_group(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
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

    # ------------------------------------------------------------------
    # Plot 5 — near-zero collapse attribution per algorithm
    # ------------------------------------------------------------------

    def plot_diversity_collapse_attribution(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
        near_zero_threshold: float = 0.01,
    ) -> Path:
        """Identify which algorithms are associated with near-zero diversity collapses.

        Two metrics are shown per algorithm:

        - **Collapse rate** — fraction of all steps during which that algorithm
          was selected and diversity was below *near_zero_threshold*.
        - **Collapse onset rate** — fraction of steps where the algorithm
          *triggered* a collapse (diversity was ≥ threshold at the previous step
          but < threshold at the current step).

        Args:
            records: Transition records from :meth:`build_records`.
            output_path: Where to save the figure.
            near_zero_threshold: Diversity values below this are considered
                near-zero / collapsed.  Default 0.01.

        Returns:
            Path to the saved figure.
        """
        # ---- per-algorithm time spent in collapsed state ----
        collapsed = records[records["diversity"] < near_zero_threshold]
        total_steps = records.groupby("curr_action")["step"].count().rename("total")
        collapse_steps = collapsed.groupby("curr_action")["step"].count().rename("collapse_steps")

        algo_stats = pd.concat([total_steps, collapse_steps], axis=1).fillna(0)
        algo_stats["collapse_rate"] = algo_stats["collapse_steps"] / algo_stats["total"]

        # ---- collapse onset: transition from non-collapsed → collapsed ----
        valid = records.dropna(subset=["diversity_change"]).copy()
        valid["prev_diversity"] = valid["diversity"] - valid["diversity_change"]
        onset = valid[
            (valid["diversity"] < near_zero_threshold)
            & (valid["prev_diversity"] >= near_zero_threshold)
        ]
        onset_per_algo = onset.groupby("curr_action")["step"].count().rename("onsets")
        algo_stats = algo_stats.join(onset_per_algo).fillna({"onsets": 0})
        algo_stats["onset_rate"] = algo_stats["onsets"] / algo_stats["total"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def _hbar(ax: plt.Axes, series: pd.Series, counts: pd.Series, color: str, xlabel: str, title: str) -> None:
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

        drop_sorted = algo_stats.sort_values("collapse_rate", ascending=False)
        _hbar(
            axes[0],
            drop_sorted["collapse_rate"],
            drop_sorted["collapse_steps"],
            color="firebrick",
            xlabel=f"Fraction of steps with diversity < {near_zero_threshold}",
            title="Collapse rate per algorithm\n(time spent in near-zero state)",
        )

        onset_sorted = algo_stats.sort_values("onset_rate", ascending=False)
        _hbar(
            axes[1],
            onset_sorted["onset_rate"],
            onset_sorted["onsets"],
            color="darkorange",
            xlabel="Fraction of steps where algorithm triggers collapse",
            title="Collapse onset rate per algorithm\n(diversity enters near-zero zone)",
        )

        if collapsed.empty:
            for ax in axes:
                ax.text(
                    0.5, 0.5, "No collapses detected",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray",
                )

        fig.suptitle(
            f"Diversity collapse attribution  (threshold = {near_zero_threshold})", fontsize=13
        )
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Plot 6 — significant diversity changes per algorithm
    # ------------------------------------------------------------------

    def plot_significant_diversity_changes(
        self,
        records: pd.DataFrame,
        output_path: str | Path,
        drop_percentile: float = 5.0,
        spike_percentile: float = 95.0,
    ) -> Path:
        """Show which algorithms are disproportionately linked to large diversity swings.

        A **significant drop** is a ``diversity_change`` at or below the
        *drop_percentile*-th percentile of all changes.  A **significant spike**
        is at or above the *spike_percentile*-th percentile.

        For each algorithm the bar shows:
        ``(# significant-drop steps) / (# total steps for that algorithm)``
        and the equivalent for spikes.

        Args:
            records: Transition records from :meth:`build_records`.
            output_path: Where to save the figure.
            drop_percentile: Bottom-N percentile defining a significant drop.
                Default 5.0.
            spike_percentile: Top-N percentile defining a significant spike.
                Default 95.0.

        Returns:
            Path to the saved figure.
        """
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

        # ---- significant drops ----
        drop_sorted = algo_stats.sort_values("drop_rate", ascending=False)
        ax1 = axes[0]
        ax1.barh(drop_sorted.index.tolist(), drop_sorted["drop_rate"], color="crimson", alpha=0.8)
        mean_d = drop_sorted["drop_rate"].mean()
        ax1.axvline(mean_d, color="gray", linestyle="--", linewidth=0.8, label=f"Mean ({mean_d:.3f})")
        ax1.set_xlabel(
            f"Fraction of steps with Δdiversity ≤ {drop_thresh:.4f}\n"
            f"(bottom {drop_percentile:.0f}% of all changes)"
        )
        ax1.set_title(f"Significant drops per algorithm")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis="x")
        x_max_d = drop_sorted["drop_rate"].max() if drop_sorted["drop_rate"].max() > 0 else 1.0
        for i, (val, cnt) in enumerate(zip(drop_sorted["drop_rate"], drop_sorted["drops"])):
            ax1.text(val + 0.01 * x_max_d, i, f"n={int(cnt)}", va="center", fontsize=8)

        # ---- significant spikes ----
        spike_sorted = algo_stats.sort_values("spike_rate", ascending=False)
        ax2 = axes[1]
        ax2.barh(spike_sorted.index.tolist(), spike_sorted["spike_rate"], color="steelblue", alpha=0.8)
        mean_s = spike_sorted["spike_rate"].mean()
        ax2.axvline(mean_s, color="gray", linestyle="--", linewidth=0.8, label=f"Mean ({mean_s:.3f})")
        ax2.set_xlabel(
            f"Fraction of steps with Δdiversity ≥ {spike_thresh:.4f}\n"
            f"(top {100 - spike_percentile:.0f}% of all changes)"
        )
        ax2.set_title(f"Significant spikes per algorithm")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x")
        x_max_s = spike_sorted["spike_rate"].max() if spike_sorted["spike_rate"].max() > 0 else 1.0
        for i, (val, cnt) in enumerate(zip(spike_sorted["spike_rate"], spike_sorted["spikes"])):
            ax2.text(val + 0.01 * x_max_s, i, f"n={int(cnt)}", va="center", fontsize=8)

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

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate all diversity analysis plots and return their paths."""
        output_dir = Path(output_dir)
        records = self.build_records()

        written: dict[str, Path] = {}
        written["diversity_distributions"] = self.plot_diversity_distributions(
            records, output_dir / "diversity_distributions.png"
        )
        written["transition_heatmap"] = self.plot_transition_heatmap(
            records, output_dir / "transition_heatmap.png"
        )
        written["increase_probability"] = self.plot_increase_probability(
            records, output_dir / "increase_probability.png"
        )
        written["diversity_change_by_group"] = self.plot_diversity_change_by_group(
            records, output_dir / "diversity_change_by_group.png"
        )
        written["collapse_attribution"] = self.plot_diversity_collapse_attribution(
            records, output_dir / "collapse_attribution.png"
        )
        written["significant_changes"] = self.plot_significant_diversity_changes(
            records, output_dir / "significant_changes.png"
        )
        return written


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------


def generate_diversity_report(
    actions_path: str | Path,
    diversity_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Convenience wrapper for script and library usage."""
    actions_path = Path(actions_path)
    if output_dir is None:
        output_dir = REPO_ROOT / "analysis_plots" / actions_path.stem / "diversity"

    analyser = DiversityAnalyser.from_jsonl_pair(actions_path, diversity_path)
    return analyser.generate_report(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse how algorithm-switching decisions affect population diversity."
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=DEFAULT_ACTIONS_PATH,
        help=f"Actions JSONL file (default: {DEFAULT_ACTIONS_PATH.name})",
    )
    parser.add_argument(
        "--diversity",
        type=Path,
        default=DEFAULT_DIVERSITY_PATH,
        help=f"Diversity JSONL file (default: {DEFAULT_DIVERSITY_PATH.name})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    resolved_output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "analysis_plots" / args.actions.stem / "diversity"
    )

    outputs = generate_diversity_report(
        actions_path=args.actions,
        diversity_path=args.diversity,
        output_dir=resolved_output_dir,
    )

    print(f"Saved diversity analysis report to: {resolved_output_dir}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
