"""Plotting utilities for analysis results."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from dynamicalgorithmselection.analysis.metrics import compute_ERT_rank
from dynamicalgorithmselection.analysis.preprocessing import extract_cdb
from itertools import permutations

SINGLE_ALGO_BASELINES = ("G3PCX", "LMCMAES", "SPSO")
SINGLE_ALGO_LINESTYLES = {
    "G3PCX": (0, (6, 2)),
    "LMCMAES": (0, (1, 1.5)),
    "SPSO": (0, (5, 1.5, 1, 1.5)),
}
SINGLE_ALGO_COLORS = {
    "G3PCX": "#000000",
    "LMCMAES": "#8B1A1A",
    "SPSO": "#4B0082",
}


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


def plot_aocc_by_cdb_per_dimension(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """AOCC vs CDB with one subplot per dimension, overlaying single-algo baselines.

    Generates one figure per CV mode (LOIO, LOPO). Each figure has one subplot
    per dimension showing the portfolio DAS AOCC curve across CDB values plus
    horizontal reference lines for the single-algorithm baselines (G3PCX,
    LMCMAES, SPSO) at that dimension.
    """
    row_filter = lambda n: "MULTIDIMENSIONAL" not in n  # noqa: E731
    das_color = "#1f77b4"
    random_color = "#2ca02c"

    for cv_mode in ("LOIO", "LOPO"):
        fig, axes = plt.subplots(
            1, len(dims), figsize=(4.4 * len(dims), 5.4), sharey=False
        )
        if len(dims) == 1:
            axes = [axes]

        for ax, dim in zip(axes, dims):
            df = datasets.get(dim, {}).get(f"aocc_{cv_mode}")
            if df is None:
                ax.set_visible(False)
                continue

            matching = [
                name
                for name in df.index
                if any(("_".join(i) in name) for i in permutations(portfolio))
                and row_filter(name)
                and all(nc not in name for nc in NON_COMPARED)
            ]
            pairs = sorted(
                (
                    (extract_cdb(name), float(df.loc[name].mean()))
                    for name in matching
                    if extract_cdb(name) is not None
                ),
                key=lambda p: p[0],
            )
            if pairs:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                ax.plot(xs, ys, marker="o", color=das_color, label="RL-Exp-DAS")

            random_rows = [
                name
                for name in df.index
                if "RANDOM" in name
                and "RANDOM_DAS" not in name
                and "MULTIDIMENSIONAL" not in name
            ]
            random_pairs = sorted(
                (
                    (extract_cdb(name), float(df.loc[name].mean()))
                    for name in random_rows
                    if extract_cdb(name) is not None
                ),
                key=lambda p: p[0],
            )
            if random_pairs:
                rxs = [p[0] for p in random_pairs]
                rys = [p[1] for p in random_pairs]
                ax.plot(
                    rxs,
                    rys,
                    marker="s",
                    linestyle="--",
                    color=random_color,
                    alpha=0.9,
                    label="Exp Random-AS",
                )

            for algo in SINGLE_ALGO_BASELINES:
                row_name = f"BASELINES_baselines_{algo}_{dim}"
                if row_name not in df.index:
                    continue
                ax.axhline(
                    float(df.loc[row_name].mean()),
                    color=SINGLE_ALGO_COLORS[algo],
                    linestyle=SINGLE_ALGO_LINESTYLES[algo],
                    alpha=0.9,
                    linewidth=1.6,
                    label=algo,
                )

            ax.set_title(f"D={dim}", fontsize=16)
            ax.set_xlabel("CDB", fontsize=13)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(
            f"AOCC (mean over problems) — {cv_mode}", fontsize=13
        )

        # Single shared legend for the whole figure (dedup across subplots)
        seen: dict[str, object] = {}
        for ax in axes:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl not in seen:
                    seen[lbl] = h
        if seen:
            fig.legend(
                handles=list(seen.values()),
                labels=list(seen.keys()),
                loc="lower center",
                ncol=len(seen),
                bbox_to_anchor=(0.5, -0.02),
                frameon=True,
                fontsize=12,
            )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_dir / f"aocc_by_cdb_per_dim_{cv_mode}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


BBOB_GROUPS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("Separable", (1, 5)),
    ("Low/mod. cond.", (6, 9)),
    ("High cond. unimodal", (10, 14)),
    ("Multimodal, struct.", (15, 19)),
    ("Multimodal, weak str.", (20, 24)),
)


def _columns_for_group(columns, fn_range: tuple[int, int]) -> list[str]:
    lo, hi = fn_range
    result = []
    for c in columns:
        parts = c.split("_")
        if len(parts) < 2 or not parts[1].startswith("f"):
            continue
        try:
            fn = int(parts[1][1:])
        except ValueError:
            continue
        if lo <= fn <= hi:
            result.append(c)
    return result


def plot_aocc_by_cdb_per_dimension_and_group(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """AOCC vs CDB split by BBOB function group (cols) and dimension (rows).

    One figure per CV mode. Each cell shows the RL-Exp-DAS curve, Exp Random-AS
    curve, and the single-algorithm horizontal baselines, restricted to
    problems in that BBOB function group.
    """
    row_filter = lambda n: "MULTIDIMENSIONAL" not in n  # noqa: E731
    das_color = "#1f77b4"
    random_color = "#2ca02c"

    for cv_mode in ("LOIO", "LOPO"):
        n_rows, n_cols = len(dims), len(BBOB_GROUPS)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.6 * n_cols, 3.0 * n_rows),
            sharex=True,
            squeeze=False,
        )

        for row, dim in enumerate(dims):
            df = datasets.get(dim, {}).get(f"aocc_{cv_mode}")
            for col, (group_name, fn_range) in enumerate(BBOB_GROUPS):
                ax = axes[row][col]

                if df is None:
                    ax.set_visible(False)
                    continue

                group_cols = _columns_for_group(df.columns, fn_range)
                if not group_cols:
                    ax.set_visible(False)
                    continue

                sub = df[group_cols]

                matching = [
                    name
                    for name in sub.index
                    if any(("_".join(i) in name) for i in permutations(portfolio))
                    and row_filter(name)
                    and all(nc not in name for nc in NON_COMPARED)
                ]
                pairs = sorted(
                    (
                        (extract_cdb(name), float(sub.loc[name].mean()))
                        for name in matching
                        if extract_cdb(name) is not None
                    ),
                    key=lambda p: p[0],
                )
                if pairs:
                    xs = [p[0] for p in pairs]
                    ys = [p[1] for p in pairs]
                    ax.plot(xs, ys, marker="o", color=das_color, label="RL-Exp-DAS")

                random_rows = [
                    name
                    for name in sub.index
                    if "RANDOM" in name
                    and "RANDOM_DAS" not in name
                    and "MULTIDIMENSIONAL" not in name
                ]
                random_pairs = sorted(
                    (
                        (extract_cdb(name), float(sub.loc[name].mean()))
                        for name in random_rows
                        if extract_cdb(name) is not None
                    ),
                    key=lambda p: p[0],
                )
                if random_pairs:
                    rxs = [p[0] for p in random_pairs]
                    rys = [p[1] for p in random_pairs]
                    ax.plot(
                        rxs,
                        rys,
                        marker="s",
                        linestyle="--",
                        color=random_color,
                        alpha=0.9,
                        label="Exp Random-AS",
                    )

                for algo in SINGLE_ALGO_BASELINES:
                    row_name = f"BASELINES_baselines_{algo}_{dim}"
                    if row_name not in sub.index:
                        continue
                    ax.axhline(
                        float(sub.loc[row_name].mean()),
                        color=SINGLE_ALGO_COLORS[algo],
                        linestyle=SINGLE_ALGO_LINESTYLES[algo],
                        alpha=0.9,
                        linewidth=1.4,
                        label=algo,
                    )

                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=14)
                if row == 0:
                    ax.set_title(
                        f"{group_name}\n(f{fn_range[0]}–f{fn_range[1]})",
                        fontsize=16,
                    )
                if row == n_rows - 1:
                    ax.set_xlabel("CDB", fontsize=16)
                if col == 0:
                    ax.set_ylabel(f"D={dim}", fontsize=16)

        seen: dict[str, object] = {}
        for row_axes in axes:
            for ax in row_axes:
                if not ax.get_visible():
                    continue
                for h, lbl in zip(*ax.get_legend_handles_labels()):
                    if lbl not in seen:
                        seen[lbl] = h
        if seen:
            fig.legend(
                handles=list(seen.values()),
                labels=list(seen.keys()),
                loc="lower center",
                ncol=len(seen),
                bbox_to_anchor=(0.5, -0.01),
                frameon=True,
                fontsize=14,
            )

        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_dir / f"aocc_by_cdb_per_dim_group_{cv_mode}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


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


def _collect_rl_and_random_by_cdb(
    df: pd.DataFrame,
    portfolio: list[str],
    predicate,
) -> tuple[dict[float, list[float]], dict[float, list[float]]]:
    """Collect per-CDB AOCC values for RL-Exp-DAS and Exponential-Random rows."""
    rl_by_cdb: dict[float, list[float]] = {}
    random_by_cdb: dict[float, list[float]] = {}
    for exp_name, row in df.iterrows():
        cdb = extract_cdb(exp_name)
        if cdb is None or cdb == 1.0:
            continue
        if not predicate(exp_name):
            continue
        if (
            any("_".join(i) in exp_name for i in permutations(portfolio))
            and "_PG_" in exp_name
        ):
            rl_by_cdb.setdefault(cdb, []).extend(row.dropna().tolist())
        elif "RANDOM" in exp_name and "RANDOM_DAS" not in exp_name:
            random_by_cdb.setdefault(cdb, []).extend(row.dropna().tolist())
    return rl_by_cdb, random_by_cdb


def plot_rl_exp_das_vs_random(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Violin plots comparing RL-Exponential-DAS vs Exponential-Random per CDB.

    For each dimension and CV mode (LOIO and LOPO), generates a violin plot
    showing the AOCC distribution of the RL policy-gradient agent next to the
    random baseline for each shared CDB value. Only CDB > 1.0 experiments are
    included (CDB=1.0 is the non-exponential RL-DAS baseline and has no
    matching Exponential-Random counterpart).

    Output filenames: ``rl_exp_das_vs_random_dim{dim}_{cv_mode}.png``.
    """
    for cv_mode in ("LOIO", "LOPO"):
        predicate_standard = lambda n: "MULTIDIMENSIONAL" not in n  # noqa: E731

        for dim in dims:
            metric_key = f"aocc_{cv_mode}"
            if metric_key not in datasets.get(dim, {}):
                continue

            df = datasets[dim][metric_key]
            rl_by_cdb, random_by_cdb = _collect_rl_and_random_by_cdb(
                df, portfolio, predicate_standard
            )
            common_cdbs = sorted(set(rl_by_cdb) & set(random_by_cdb))
            if not common_cdbs:
                continue

            fig, ax = plt.subplots(figsize=(max(8, 3 * len(common_cdbs)), 6))
            fig.suptitle(
                f"RL-Exponential-DAS vs Exponential-Random"
                f" — Dimension {dim} ({cv_mode})",
                fontsize=14,
            )

            labels: list[str] = []
            violin_data: list[list[float]] = []
            colors: list[str] = []
            for cdb in common_cdbs:
                labels.append(f"RL-Exp-DAS\n(CDB={cdb:g})")
                violin_data.append(rl_by_cdb[cdb])
                colors.append("tab:orange")
                labels.append(f"Exp-Random\n(CDB={cdb:g})")
                violin_data.append(random_by_cdb[cdb])
                colors.append("tab:green")

            parts = ax.violinplot(
                violin_data, positions=range(len(labels)), showmeans=True
            )
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.8)

            for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
                if partname in parts:
                    parts[partname].set_edgecolor("black")
                    parts[partname].set_linewidth(1.5)

            ax.set_ylabel("AOCC distribution over problems")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            _save_and_close(
                fig, save_dir, f"rl_exp_das_vs_random_dim{dim}_{cv_mode}.png"
            )


def plot_rl_vs_random_with_significance(
    datasets: dict[int, dict[str, pd.DataFrame]],
    wilcoxon_table: pd.DataFrame,
    portfolio: list[str],
    save_dir: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Combined mean-AOCC lines + Wilcoxon heatmap for RL-Exp-DAS vs Exp-Random.

    Generates one figure per CV mode family (LOIO standard, LOPO standard,
    LOPO multi). Each figure is 1 column × 2 rows:
    - Top: mean AOCC per dimension for RL-Exp-DAS (solid) and Exponential-Random
      (dashed) across CDB values.
    - Bottom: rank-biserial effect size heatmap with Holm-adjusted significance
      markers (rows = dimensions, columns = CDB values).

    Output filenames:
      ``rl_vs_random_with_significance_aocc_LOIO.png``
      ``rl_vs_random_with_significance_aocc_LOPO.png``
      ``rl_vs_random_with_significance_aocc_multi_LOPO.png``
    """
    metric = "aocc"
    panels = [
        ("LOIO", "standard", "rl_vs_random_with_significance_aocc_LOIO.png"),
        ("LOPO", "standard", "rl_vs_random_with_significance_aocc_LOPO.png"),
        ("LOPO", "multi", "rl_vs_random_with_significance_aocc_multi_LOPO.png"),
    ]

    for cv_mode, exp_type, filename in panels:
        fig, (ax_line, ax_heat) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.4, 1.0]}
        )

        if exp_type == "multi":
            predicate = lambda n: "MULTIDIMENSIONAL" in n  # noqa: E731
        else:
            predicate = lambda n: "MULTIDIMENSIONAL" not in n  # noqa: E731

        for dim in dims:
            df = datasets.get(dim, {}).get(f"{metric}_{cv_mode}")
            if df is None:
                continue

            rl_by_cdb, random_by_cdb = _collect_rl_and_random_by_cdb(
                df, portfolio, predicate
            )
            common_cdbs = sorted(set(rl_by_cdb) & set(random_by_cdb))
            if not common_cdbs:
                continue

            color = DIM_COLORS.get(dim)
            rl_means = [float(pd.Series(rl_by_cdb[c]).mean()) for c in common_cdbs]
            rnd_means = [float(pd.Series(random_by_cdb[c]).mean()) for c in common_cdbs]

            ax_line.plot(
                common_cdbs,
                rl_means,
                marker="o",
                linestyle="-",
                label=f"RL D={dim}",
                color=color,
            )
            ax_line.plot(
                common_cdbs,
                rnd_means,
                marker="s",
                linestyle="--",
                label=f"Random D={dim}",
                color=color,
                alpha=0.6,
            )

        ax_line.set_xlabel("CDB")
        ax_line.set_ylabel(f"{metric.upper()} (mean over problems)")
        ax_line.legend(fontsize=7, ncol=2)
        ax_line.grid(True, alpha=0.3)

        sub = wilcoxon_table[
            (wilcoxon_table.cv_mode == cv_mode)
            & (wilcoxon_table.experiment_type == exp_type)
        ]
        if sub.empty:
            ax_heat.set_visible(False)
        else:
            pivot_eff = sub.pivot(index="dim", columns="cdb", values="rank_biserial")
            pivot_p = sub.pivot(index="dim", columns="cdb", values="p_value_holm")
            pivot_eff = pivot_eff.reindex(sorted(pivot_eff.columns), axis=1)
            pivot_p = pivot_p.reindex(sorted(pivot_p.columns), axis=1)
            pivot_eff = pivot_eff.reindex(sorted(pivot_eff.index))
            pivot_p = pivot_p.reindex(sorted(pivot_p.index))

            ax_heat.imshow(
                pivot_eff.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
            )
            for i in range(pivot_eff.shape[0]):
                for j in range(pivot_eff.shape[1]):
                    eff = pivot_eff.iloc[i, j]
                    mark = _significance_marker(pivot_p.iloc[i, j])
                    color = (
                        "white"
                        if not pd.isna(eff) and abs(eff) > 0.55
                        else "black"
                    )
                    ax_heat.text(
                        j, i, mark, ha="center", va="center", fontsize=10, color=color
                    )

            ax_heat.set_xticks(range(len(pivot_eff.columns)))
            ax_heat.set_xticklabels([f"{c:g}" for c in pivot_eff.columns])
            ax_heat.set_yticks(range(len(pivot_eff.index)))
            ax_heat.set_yticklabels([f"D={d}" for d in pivot_eff.index])
            ax_heat.set_xlabel("CDB (RL-Exp-DAS vs Exponential-Random)")

        fig.tight_layout()
        _save_and_close(fig, save_dir, filename)


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
