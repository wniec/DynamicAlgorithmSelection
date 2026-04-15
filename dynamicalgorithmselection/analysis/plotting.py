"""Plotting utilities for analysis results."""

from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamicalgorithmselection.analysis.metrics import compute_ERT_rank
from dynamicalgorithmselection.analysis.preprocessing import extract_cdb


# Globals cleanly defined at the top
DIM_COLORS = {2: "tab:blue", 3: "tab:orange", 5: "tab:green", 10: "tab:red"}
NON_COMPARED = ["RANDOM", "MULTIDIMENSIONAL", "REWARD"]


def _plot_lines_by_dimension(
    ax: plt.Axes,
    dims: tuple[int, ...],
    data_extractor: Callable[[int], list[tuple[float, float]]],
) -> None:
    """Helper to extract data, sort by CDB, and plot lines per dimension.

    Args:
        ax: The matplotlib Axes object to plot on.
        dims: The tuple of dimensions to iterate over.
        data_extractor: A callback function that takes an integer dimension
            and returns a list of (cdb, metric_value) tuples.
    """
    for dim in dims:
        dim_data = data_extractor(dim)
        if not dim_data:
            continue

        # Sort strictly by CDB (index 0) to prevent Python from breaking ties
        # by sorting the Y values, which creates false vertical trends.
        pairs = sorted([(key, val) for key, val in dim_data], key=lambda x: x[0])
        xs, ys = [i[0] for i in pairs], [i[1] for i in pairs]

        ax.plot(
            xs,
            ys,
            marker="o",
            label=f"DIM {dim}",
            color=DIM_COLORS.get(dim),
        )


def plot_cdb_impact(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: str,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Plot AUOC/AOCC vs CDB for standard and multidimensional training.

    Args:
        datasets: Dictionary of DataFrames segmented by dimension and metric.
        portfolio: Portfolio name to filter experiments by.
        dims: Dimensions to plot.
    """
    # 1. Standard LOIO / LOPO Plots
    for metric, cv_mode in product(("auoc", "aocc"), ("LOIO", "LOPO")):
        fig, ax = plt.subplots(figsize=(8, 5))

        def extract_standard_data(dim: int) -> list[tuple[float, float]]:
            df = datasets[dim][f"{metric}_{cv_mode}"]
            df = (df - df.mean()) / df.std()
            matching = [
                name
                for name in df.index
                if portfolio in name and all(nc not in name for nc in NON_COMPARED)
            ]

            # Using list comprehension prevents overwriting duplicate CDB values
            return [
                (extract_cdb(name), float(df.loc[name].mean()))
                for name in matching
                if extract_cdb(name) is not None
            ]

        _plot_lines_by_dimension(ax, dims, extract_standard_data)

        ax.set_xlabel("CDB")
        ax.set_ylabel(f"{metric.upper()} (mean over problems)")
        ax.set_title(f"CDB impact — {portfolio} — {cv_mode}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    # 2. Multidimensional Plots (LOPO only)
    cv_mode = "LOPO"
    for metric in ("auoc", "aocc"):
        fig, ax = plt.subplots(figsize=(8, 5))

        def extract_multi_data(dim: int) -> list[tuple[float, float]]:
            df = datasets[dim][f"{metric}_{cv_mode}"]
            df = (df - df.mean()) / df.std()
            matching = [
                name
                for name in df.index
                if portfolio in name and all(nc not in name for nc in NON_COMPARED)
            ]

            return [
                (extract_cdb(name), float(df.loc[name].mean()))
                for name in matching
                if extract_cdb(name) is not None
            ]

        _plot_lines_by_dimension(ax, dims, extract_multi_data)

        ax.set_xlabel("CDB")
        ax.set_ylabel(f"{metric.upper()} (mean over problems)")
        ax.set_title(f"CDB impact — {portfolio} — {cv_mode} Multidimensional training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cdb_impact_comparison(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: str,
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
        if portfolio in name:
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
        metric_key = f"auoc_{cv_mode}"

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
        plt.show()


def plot_ert_impact(
    datasets: dict[int, pd.DataFrame],
    portfolio: str,
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
                if portfolio in name
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
        plt.show()

    # 2. Multidimensional Plots (LOPO only)
    cv_mode = "LOPO"
    fig, ax = plt.subplots(figsize=(8, 5))

    def extract_multi_ert_data(dim: int) -> list[tuple[float, float]]:
        df = datasets[dim]
        matching = [
            name
            for name in df.index
            if portfolio in name and "MULTIDIMENSIONAL" in name and cv_mode in name
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
    ax.set_title(f"CDB impact — {portfolio} — {cv_mode} MULTIDIMENSIONAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
