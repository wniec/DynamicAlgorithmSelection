"""Plotting utilities for analysis results."""

from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from dynamicalgorithmselection.analysis.preprocessing import extract_cdb


def plot_cdb_impact(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: str,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Plot AUOC vs CDB for each CV mode (LOIO / LOPO).

    For the given *portfolio*, finds matching experiments across all
    dimensions, extracts the CDB value from each experiment name, and
    produces two plots (one per CV mode) with dimensions shown in
    different colours.
    """
    dim_colors = {2: "tab:blue", 3: "tab:orange", 5: "tab:green", 10: "tab:red"}
    NON_COMPARED = ["RANDOM", "MULTIDIMENSIONAL", "REWARD"]
    for metric, cv_mode in product(("auoc", "aocc"), ("LOIO", "LOPO")):
        fig, ax = plt.subplots(figsize=(8, 5))

        for dim in dims:
            df = datasets[dim][f"{metric}_{cv_mode}"]
            df = (df - df.mean()) / df.std()
            matching = [
                name
                for name in df.index
                if portfolio in name and all(i not in name for i in NON_COMPARED)
            ]
            if not matching:
                continue

            cdb_vals: list[float] = []
            auoc_means: list[float] = []
            for name in matching:
                cdb = extract_cdb(name)
                if cdb is None:
                    continue
                cdb_vals.append(cdb)
                auoc_means.append(df.loc[name].mean())

            if not cdb_vals:
                continue

            # Sort by CDB for a clean line plot
            pairs = sorted(zip(cdb_vals, auoc_means))
            xs, ys = zip(*pairs)

            ax.plot(
                xs,
                ys,
                marker="o",
                label=f"DIM {dim}",
                color=dim_colors.get(dim),
            )

        ax.set_xlabel("CDB")
        ax.set_ylabel(f"{metric.upper()} (mean over problems)")
        ax.set_title(f"CDB impact — {portfolio} — {cv_mode}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


def plot_ert_impact(
    datasets: dict[int, pd.Series],
    portfolio: str,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> None:
    """Plot AUOC vs CDB for each CV mode (LOIO / LOPO).

    For the given *portfolio*, finds matching experiments across all
    dimensions, extracts the CDB value from each experiment name, and
    produces two plots (one per CV mode) with dimensions shown in
    different colours.
    """
    dim_colors = {2: "tab:blue", 3: "tab:orange", 5: "tab:green", 10: "tab:red"}
    NON_COMPARED = ["RANDOM", "MULTIDIMENSIONAL", "REWARD"]
    for cv_mode in ("LOIO", "LOPO"):
        fig, ax = plt.subplots(figsize=(8, 5))

        for dim in dims:
            df = datasets[dim]
            matching = [
                name
                for name in df.index
                if portfolio in name
                and all(i not in name for i in NON_COMPARED)
                and cv_mode in name
            ]
            if not matching:
                continue

            cdb_vals: list[float] = []
            auoc_means: list[float] = []
            for name in matching:
                cdb = extract_cdb(name)
                if cdb is None:
                    continue
                cdb_vals.append(cdb)
                auoc_means.append(df.loc[name].mean())

            if not cdb_vals:
                continue

            # Sort by CDB for a clean line plot
            pairs = sorted(zip(cdb_vals, auoc_means))
            xs, ys = zip(*pairs)

            ax.plot(
                xs,
                ys,
                marker="o",
                label=f"DIM {dim}",
                color=dim_colors.get(dim),
            )

        ax.set_xlabel("CDB")
        ax.set_ylabel(f"(mean ERT rank over problems)")
        ax.set_title(f"CDB impact — {portfolio} — {cv_mode}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
