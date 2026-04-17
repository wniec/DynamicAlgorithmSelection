"""Run both analysis pipelines on results_cleaned/ and print summaries."""

import argparse
import re
from pathlib import Path

import pandas as pd

from dynamicalgorithmselection.analysis.latex_utils import get_output_latex
from dynamicalgorithmselection.analysis.loading import (
    load_experiment_results,
    load_ert_htmls,
)
from dynamicalgorithmselection.analysis.metrics import (
    compute_solve_rate,
    extract_metric,
    parse_ert_from_html,
    compute_ERT_rank,
)
from dynamicalgorithmselection.analysis.plotting import (
    plot_ert_impact,
    plot_cdb_impact_comparison,
    plot_cdb_impact_with_significance,
)
from dynamicalgorithmselection.analysis.preprocessing import (
    aggregate_over_seeds,
    split_ert_by_dimension,
    split_results_by_dimension,
)
from dynamicalgorithmselection.analysis.statistical_tests import (
    compute_cdb_wilcoxon_table,
)

DATA_DIR = Path(".")
DIMS = (2, 3, 5, 10)
EXTRA_BASELINES = [
    f"BASELINES_baselines_{name}"
    for name in ("MADDE", "JDE21", "NL_SHADE_RSP", "best", "worst")
]

# Global list to store Series/DataFrames for the final joint table
LOPO_TABLES = []
LOIO_TABLES = []


def strip_dim_from_index(data_obj, dim: int):
    """
    Strips dimension-specific suffixes from the pandas index so
    cross-dimensional joins align correctly.
    Handles: '_DIM{dim}' anywhere, and '_{dim}' at the end of the string.
    """
    # Create a copy to avoid SettingWithCopy warnings just in case
    data_obj = data_obj.copy()
    data_obj.index = data_obj.index.str.replace(
        f"_DIM{dim}", "", regex=False
    ).str.replace(rf"_{dim}$", "", regex=True)
    return data_obj


def run_results_pipeline(
    portfolio: str = "G3PCX_LMCMAES_SPSO_",
    plots_dir: Path = Path("plots"),
    stats_path: Path = Path("plots/cdb_wilcoxon.csv"),
) -> None:
    print("=" * 60)
    print("RESULTS PIPELINE")
    print("=" * 60)
    results_dir = DATA_DIR / "results"
    results = load_experiment_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    final_fitness = extract_metric(results, "final_fitness")
    aocc = extract_metric(results, "aocc")
    print(
        f"Raw shapes — final_fitness: {final_fitness.shape}, AOCC: {aocc.shape}"
    )

    ff_agg = aggregate_over_seeds(final_fitness)
    aocc_agg = aggregate_over_seeds(aocc)

    print(
        f"After seed aggregation — final_fitness: {ff_agg.shape}, AOCC: {aocc_agg.shape}"
    )

    datasets = split_results_by_dimension(
        ff_agg, aocc_agg, dims=DIMS, extra_baselines=EXTRA_BASELINES
    )

    for dim in DIMS:
        print(f"\n--- Dimension {dim} ---")

        # We will extract both LOPO and LOIO for all metrics
        scenarios = ["LOPO", "LOIO"]
        tables = (LOPO_TABLES, LOIO_TABLES)

        for table, scenario in zip(tables, scenarios):
            # Final Fitness
            ff_key = f"final_fitness_{scenario}"
            if ff_key in datasets[dim]:
                means_ff = datasets[dim][ff_key].mean(axis=1)
                means_ff.name = f"FF_DIM{dim}"
                table.append(strip_dim_from_index(means_ff, dim))

            # AOCC
            aocc_key = f"aocc_{scenario}"
            if aocc_key in datasets[dim]:
                means_aocc = datasets[dim][aocc_key].mean(axis=1)
                means_aocc.name = f"AOCC_DIM{dim}"
                table.append(strip_dim_from_index(means_aocc, dim))

    # --- CDB impact plots for the selected portfolio ---
    plot_cdb_impact_comparison(datasets, portfolio, save_dir=plots_dir, dims=DIMS)

    # --- Wilcoxon signed-rank (one-sided) vs CDB=1.0 baseline ---
    stats_table = compute_cdb_wilcoxon_table(
        datasets, portfolio, save_path=stats_path, dims=DIMS
    )
    print(
        f"\nWilcoxon table: {stats_table.shape[0]} comparisons -> {stats_path.resolve()}"
    )
    plot_cdb_impact_with_significance(
        datasets, stats_table, portfolio, save_dir=plots_dir, dims=DIMS
    )


def run_ert_pipeline(
    portfolio: str = "G3PCX_LMCMAES_SPSO_",
    plots_dir: Path = Path("plots"),
) -> None:
    print("\n" + "=" * 60)
    print("ERT PIPELINE")
    print("=" * 60)

    htmls = load_ert_htmls(DATA_DIR / "ppdata")
    print(f"Loaded {len(htmls)} HTML files")

    all_ert: list[dict[str, object]] = []
    for name, html in htmls.items():
        data: dict[str, object] = parse_ert_from_html(html)
        seed_match = re.search(r"_SEED(\d+)", name)
        data["base_name"] = (
            name[: seed_match.start()].replace(".html", "") if seed_match else name
        )
        all_ert.append(data)

    df_ert = pd.DataFrame(all_ert).groupby("base_name").mean(numeric_only=True)
    print(f"ERT DataFrame: {df_ert.shape}")

    ert_datasets = split_ert_by_dimension(df_ert, dims=DIMS)

    for dim in DIMS:
        print(f"\n--- Dimension {dim} ---")
        print(f"  ERT shape: {ert_datasets[dim].shape}")

    solve_rates = compute_solve_rate(ert_datasets)
    ert_rankings = compute_ERT_rank(ert_datasets)

    for dim in DIMS:
        print(f"\n  Solve rate dim={dim} (top 5):")
        for name, val in solve_rates[dim].tail(5).items():
            print(f"    {val:.3f}  {name}")

        # Extract ERT ranking, rename, clean index, and append
        ert_rank = ert_rankings[dim]
        ert_rank = ert_rank.add_suffix(f"_ERT_DIM{dim}")

        # GLOBAL_TABLES.append(strip_dim_from_index(ert_rank, dim))

    plot_ert_impact(ert_datasets, portfolio, save_dir=plots_dir, dims=DIMS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis pipelines")
    parser.add_argument(
        "--portfolio",
        default="G3PCX_LMCMAES_SPSO",
        help="Portfolio name to analyse CDB impact for (default: G3PCX_LMCMAES_SPSO)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        type=Path,
        help="Directory where all generated plots are saved (default: plots/)",
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        type=Path,
        help="CSV path for the CDB Wilcoxon table (default: <plots-dir>/cdb_wilcoxon.csv)",
    )
    args = parser.parse_args()

    stats_path = args.stats_path or (args.plots_dir / "cdb_wilcoxon.csv")

    run_results_pipeline(
        portfolio=args.portfolio + "_",
        plots_dir=args.plots_dir,
        stats_path=stats_path,
    )
    run_ert_pipeline(plots_dir=args.plots_dir)

    get_output_latex(LOIO_TABLES, LOPO_TABLES)
    print(f"\nAll plots saved to: {args.plots_dir.resolve()}")
