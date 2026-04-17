"""Plotting utilities for analysis results."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from dynamicalgorithmselection.analysis.metrics import compute_ERT_rank
from dynamicalgorithmselection.analysis.preprocessing import extract_cdb
from itertools import permutations


def _save_and_close(fig: plt.Figure, save_dir: Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Globals cleanly defined at the top
DIM_COLORS = {2: "tab:blue", 3: "tab:orange", 5: "tab:green", 10: "tab:red"}
NON_COMPARED = ["RANDOM", "MULTIDIMENSIONAL", "REWARD"]


def _plot_lines_by_dimension(
    ax: plt.Axes,
    dims: tuple[int, ...],
    data_extractor: Callable[[int], list[tuple[float, float]]],
) -> None:
    """Helper to extract data, sort by CDB, and plot lines per dimension."""
    for dim in dims:
        dim_data = data_extractor(dim)
        if not dim_data:
            continue

        pairs = sorted(dim_data, key=lambda x: x[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        ax.plot(xs, ys, marker="o", label=f"DIM {dim}", color=DIM_COLORS.get(dim))


def _significance_marker(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_wilcoxon_heatmap(
    table: pd.DataFrame,
    save_dir: Path,
    filename: str = "cdb_wilcoxon_heatmap.png",
) -> None:
    """Heatmap of rank-biserial effect size for each CDB vs baseline.

    One subplot per (cv_mode, experiment_type); rows are dimensions,
    columns are CDB values (excluding the baseline). Cell color encodes
    rank-biserial correlation; annotation encodes Holm-adjusted significance.
    """
    if table.empty:
        return

    families = sorted(
        {(r.cv_mode, r.experiment_type) for r in table.itertuples(index=False)}
    )
    if not families:
        return

    fig, axes = plt.subplots(
        1, len(families), figsize=(4.2 * len(families), 3.6), squeeze=False
    )
    axes = axes[0]
    im = None

    for ax, (cv_mode, exp_type) in zip(axes, families):
        sub = table[(table.cv_mode == cv_mode) & (table.experiment_type == exp_type)]
        pivot_eff = sub.pivot(index="dim", columns="cdb", values="rank_biserial")
        pivot_p = sub.pivot(index="dim", columns="cdb", values="p_value_holm")

        pivot_eff = pivot_eff.reindex(sorted(pivot_eff.columns), axis=1)
        pivot_p = pivot_p.reindex(sorted(pivot_p.columns), axis=1)

        im = ax.imshow(
            pivot_eff.values,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )

        for i, _dim in enumerate(pivot_eff.index):
            for j, _cdb in enumerate(pivot_eff.columns):
                eff = pivot_eff.iloc[i, j]
                mark = _significance_marker(pivot_p.iloc[i, j])
                color = "white" if not pd.isna(eff) and abs(eff) > 0.55 else "black"
                ax.text(j, i, mark, ha="center", va="center", fontsize=10, color=color)

        ax.set_xticks(range(len(pivot_eff.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot_eff.columns])
        ax.set_yticks(range(len(pivot_eff.index)))
        ax.set_yticklabels([f"DIM {d}" for d in pivot_eff.index])
        ax.set_xlabel("CDB (vs. CDB=1.0)")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("rank-biserial effect size")

    _save_and_close(fig, save_dir, filename)


def _plot_combined_cdb_panel(
    ax_line: plt.Axes,
    ax_heat: plt.Axes,
    datasets: dict[int, dict[str, pd.DataFrame]],
    wilcoxon_table: pd.DataFrame,
    portfolio: list[str],
    dims: tuple[int, ...],
    cv_mode: str,
    exp_type: str,
    metric: str,
) -> None:
    if exp_type == "multi":
        row_filter = lambda n: "MULTIDIMENSIONAL" in n  # noqa: E731
    else:
        row_filter = lambda n: "MULTIDIMENSIONAL" not in n  # noqa: E731

    def extract_data(dim: int) -> list[tuple[float, float]]:
        df = datasets[dim][f"{metric}_{cv_mode}"]
        matching = [
            name
            for name in df.index
            if any(("_".join(i) in name) for i in permutations(portfolio))
            and row_filter(name)
            and all(nc not in name for nc in NON_COMPARED)
        ]
        return [
            (extract_cdb(name), float(df.loc[name].mean()))
            for name in matching
            if extract_cdb(name) is not None
        ]

    _plot_lines_by_dimension(ax_line, dims, extract_data)
    # Use "D=X" labels instead of default "DIM X"
    handles, labels = ax_line.get_legend_handles_labels()
    labels = [lbl.replace("DIM ", "D=") for lbl in labels]
    ax_line.legend(handles, labels)
    ax_line.set_xlabel("CDB")
    ax_line.set_ylabel(f"{metric.upper()} (mean over problems)")
    ax_line.grid(True, alpha=0.3)

    sub = wilcoxon_table[
        (wilcoxon_table.cv_mode == cv_mode)
        & (wilcoxon_table.experiment_type == exp_type)
    ]
    if sub.empty:
        ax_heat.set_visible(False)
        return

    pivot_eff = sub.pivot(index="dim", columns="cdb", values="rank_biserial")
    pivot_p = sub.pivot(index="dim", columns="cdb", values="p_value_holm")
    pivot_eff = pivot_eff.reindex(sorted(pivot_eff.columns), axis=1)
    pivot_p = pivot_p.reindex(sorted(pivot_p.columns), axis=1)
    pivot_eff = pivot_eff.reindex(sorted(pivot_eff.index))
    pivot_p = pivot_p.reindex(sorted(pivot_p.index))

    ax_heat.imshow(pivot_eff.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    for i in range(pivot_eff.shape[0]):
        for j in range(pivot_eff.shape[1]):
            eff = pivot_eff.iloc[i, j]
            mark = _significance_marker(pivot_p.iloc[i, j])
            color = "white" if not pd.isna(eff) and abs(eff) > 0.55 else "black"
            ax_heat.text(j, i, mark, ha="center", va="center", fontsize=10, color=color)

    ax_heat.set_xticks(range(len(pivot_eff.columns)))
    ax_heat.set_xticklabels([f"{c:g}" for c in pivot_eff.columns])
    ax_heat.set_yticks(range(len(pivot_eff.index)))
    ax_heat.set_yticklabels([f"D={d}" for d in pivot_eff.index])
    ax_heat.set_xlabel("CDB (vs. CDB=1.0)")


def plot_cdb_impact_with_significance(
    datasets: dict[int, dict[str, pd.DataFrame]],
    wilcoxon_table: pd.DataFrame,
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Combined AOCC-vs-CDB lines + Wilcoxon heatmap, one figure per family.

    Generates three figures (LOIO standard, LOPO standard, LOPO multi). Each
    figure is 1 column x 2 rows: top shows mean AOCC across problems per
    dimension over CDB; bottom shows rank-biserial effect size vs CDB=1.0
    with Holm-adjusted significance markers.
    """
    metric = "aocc"
    panels = [
        ("LOIO", "standard", "cdb_impact_with_significance_aocc_LOIO.png"),
        ("LOPO", "standard", "cdb_impact_with_significance_aocc_LOPO.png"),
        ("LOPO", "multi", "cdb_impact_with_significance_aocc_multi_LOPO.png"),
    ]

    for cv_mode, exp_type, filename in panels:
        fig, (ax_line, ax_heat) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.4, 1.0]}
        )
        _plot_combined_cdb_panel(
            ax_line,
            ax_heat,
            datasets,
            wilcoxon_table,
            portfolio,
            dims,
            cv_mode,
            exp_type,
            metric,
        )
        fig.tight_layout()
        _save_and_close(fig, save_dir, filename)


def plot_cdb_impact_comparison(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
    cv_mode: str = "LOPO",
) -> None:
    """Plot a violin plot comparing baselines, RL-DAS, Exp-DAS (CDB=2.1), Multi Exp-DAS, and Random agents on AOCC."""

    def get_category(name: str) -> str | None:
        # Extract baselines (e.g., MADDE, JDE21, best, worst)
        if "BASELINES_baselines_" in name:
            return name.split("BASELINES_baselines_")[-1]
        if "best" in name:
            return "best case"
        if "worst" in name:
            return "worst case"
        # Extract Random agents
        if "RANDOM_CDB2.1" in name:
            return "Random-Exp (CDB=2.1)"
        if "RANDOM" in name and "CDB" not in name:
            return "Random AS"

        # Extract DAS variations
        if any("_".join(i) in name for i in permutations(portfolio)):
            if "MULTIDIMENSIONAL" in name:
                if "CDB2.1" in name:
                    return "Multi Exp-DAS (CDB=2.1)"
            else:
                if "CDB2.1" in name:
                    return "Exp-DAS (CDB=2.1)"
                # Standard RL-DAS typically relies on CDB 1.0
                if "CDB1.0" in name:
                    return "RL-DAS"

        return None

    for dim in dims:
        metric_key = f"aocc_{cv_mode}"

        # Verify we have data for this dimension and metric before creating plots
        if metric_key not in datasets.get(dim, {}):
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Agent Comparison - Dimension {dim} ({cv_mode})", fontsize=16)

        df = datasets[dim][metric_key]

        cat_data = {}
        # For a violin plot, we need the raw distribution of values instead of just the mean.
        # This gathers all problem/run scores for every experiment in that category.
        for exp_name, row in df.iterrows():
            cat = get_category(exp_name)
            if cat:
                cat_data.setdefault(cat, []).extend(row.dropna().tolist())

        # Set a standard display order
        order = [
            "RL-DAS",
            "Exp-DAS (CDB=2.1)",
            "Multi Exp-DAS (CDB=2.1)",
            "Random AS",
            "Random-Exp (CDB=2.1)",
            "MADDE",
            "JDE21",
            "NL_SHADE_RSP",
            "best",
            "worst",
        ]

        # Filter labels to those present in data with at least one value
        labels = [c for c in order if c in cat_data and len(cat_data[c]) > 0]
        labels += [c for c in cat_data if c not in labels and len(cat_data[c]) > 0]
        dataset = [cat_data[c] for c in labels]

        if not dataset:
            plt.close(fig)
            continue

        # Create violin plot
        parts = ax.violinplot(dataset, positions=range(len(labels)), showmeans=True)

        # Highlight groups by color and apply hatching for easy reading
        for i, label in enumerate(labels):
            if "Multi" in label and "DAS" in label:
                color = "tab:orange"
                hatch = "//"
            elif "DAS" in label:
                color = "tab:orange"
                hatch = ""
            elif "Random" in label:
                color = "tab:green"
                hatch = ""
            else:
                color = "tab:blue"
                hatch = ""

            pc = parts["bodies"][i]
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.8)
            if hatch:
                pc.set_hatch(hatch)

        # Style the lines within the violin plot
        for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1.5)

        ax.set_title("AOCC")
        ax.set_ylabel("Distribution of Values over problems")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        _save_and_close(fig, save_dir, f"cdb_impact_comparison_dim{dim}_{cv_mode}.png")


def plot_ert_impact(
    datasets: dict[int, pd.DataFrame],
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Plot ERT rank vs CDB for standard and multidimensional training.

    Args:
        datasets: Per-dimension ERT DataFrames/Series.
        portfolio: Portfolio name to filter experiments by.
        dims: Dimensions to plot.
    """
    # 1. Standard LOIO / LOPO Plots
    for cv_mode in ("LOIO", "LOPO"):
        fig, ax = plt.subplots(figsize=(8, 5))

        def extract_ert_data(dim: int) -> list[tuple[float, float]]:
            df = datasets[dim]
            matching = [
                name
                for name in df.index
                if any(("_".join(i) in name) for i in permutations(portfolio))
                and all(i not in name for i in NON_COMPARED)
                and cv_mode in name
            ]
            if not matching:
                return []

            matching_df = df.loc[matching]
            ert_ranks = compute_ERT_rank({dim: matching_df})[dim]

            return [
                (extract_cdb(name), float(ert_ranks.loc[name]))
                for name in matching
                if extract_cdb(name) is not None and name in ert_ranks.index
            ]

        _plot_lines_by_dimension(ax, dims, extract_ert_data)

        ax.set_xlabel("CDB")
        ax.set_ylabel("(mean ERT ranking over problems)")
        ax.set_title(f"CDB impact — {portfolio} — {cv_mode}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_and_close(fig, save_dir, f"ert_impact_{cv_mode}.png")

    # 2. Multidimensional Plots (LOPO only)
    cv_mode = "LOPO"
    fig, ax = plt.subplots(figsize=(8, 5))

    def extract_multi_ert_data(dim: int) -> list[tuple[float, float]]:
        df = datasets[dim]
        matching = [
            name
            for name in df.index
            if any(("_".join(i) in name) for i in permutations(portfolio))
            and "MULTIDIMENSIONAL" in name
            and cv_mode in name
        ]
        if not matching:
            return []

        matching_df = df.loc[matching]
        ert_ranks = compute_ERT_rank({dim: matching_df})[dim]

        return [
            (extract_cdb(name), float(ert_ranks.loc[name]))
            for name in matching
            if extract_cdb(name) is not None and name in ert_ranks.index
        ]

    _plot_lines_by_dimension(ax, dims, extract_multi_ert_data)

    ax.set_xlabel("CDB")
    ax.set_ylabel("(mean ERT ranking over MULTIDIMENSIONAL problems)")
    ax.set_title(f"CDB impact — {' '.join(portfolio)} — {cv_mode} MULTIDIMENSIONAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_and_close(fig, save_dir, f"ert_impact_multi_{cv_mode}.png")
