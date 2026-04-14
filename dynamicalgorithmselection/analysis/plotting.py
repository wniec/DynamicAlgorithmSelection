"""Plotting utilities for analysis results."""

from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
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
