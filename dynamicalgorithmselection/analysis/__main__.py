"""Run both analysis pipelines on results_cleaned/ and print summaries."""

import re
from pathlib import Path

import pandas as pd

from dynamicalgorithmselection.analysis.loading import (
    load_experiment_results,
    load_ert_htmls,
)
from dynamicalgorithmselection.analysis.metrics import (
    compute_solve_rate,
    extract_metric,
    parse_ert_from_html,
)
from dynamicalgorithmselection.analysis.plotting import plot_cdb_impact
from dynamicalgorithmselection.analysis.preprocessing import (
    aggregate_over_seeds,
    split_ert_by_dimension,
    split_results_by_dimension,
)

DATA_DIR = Path(".")
DIMS = (2, 3, 5, 10)
EXTRA_BASELINES = [
    f"BASELINES_baselines_{name}" for name in ("MADDE", "JDE21", "NL_SHADE_RSP")
]


def run_results_pipeline(
    portfolio: str = "G3PCX_LMCMAES_SPSO",
) -> None:
    print("=" * 60)
    print("RESULTS PIPELINE")
    print("=" * 60)
    results_dir = DATA_DIR / "results"
    results = load_experiment_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    auoc = extract_metric(results, "area_under_optimization_curve")
    final_fitness = extract_metric(results, "final_fitness")
    print(f"Raw shapes — AUOC: {auoc.shape}, final_fitness: {final_fitness.shape}")

    auoc_agg = aggregate_over_seeds(auoc)
    ff_agg = aggregate_over_seeds(final_fitness)
    print(
        f"After seed aggregation — AUOC: {auoc_agg.shape}, final_fitness: {ff_agg.shape}"
    )

    datasets = split_results_by_dimension(
        auoc_agg, ff_agg, dims=DIMS, extra_baselines=EXTRA_BASELINES
    )

    for dim in DIMS:
        print(f"\n--- Dimension {dim} ---")
        for key, df in datasets[dim].items():
            print(f"  {key}: {df.shape}")

        print(f"\n  AUOC LOPO ranking (top 5):")
        means = datasets[dim]["auoc_LOPO"].mean(axis=1).sort_values()
        for name, val in means.head(5).items():
            print(f"    {val:12.2f}  {name}")

        print(f"\n  Final fitness LOIO ranking (top 5):")
        means = datasets[dim]["final_fitness_LOIO"].mean(axis=1).sort_values()
        for name, val in means.head(5).items():
            print(f"    {val:12.6f}  {name}")

    # --- CDB impact plots for the selected portfolio ---
    plot_cdb_impact(datasets, portfolio, dims=DIMS)


def run_ert_pipeline() -> None:
    print("\n" + "=" * 60)
    print("ERT PIPELINE")
    print("=" * 60)

    htmls = load_ert_htmls(DATA_DIR / "ppdata")
    print(f"Loaded {len(htmls)} HTML files")

    all_ert: list[dict[str, object]] = []
    for name, html in htmls.items():
        data: dict[str, object] = parse_ert_from_html(html)
        seed_match = re.search(r"_SEED(\d+)", name)
        data["base_name"] = name[: seed_match.start()] if seed_match else name
        all_ert.append(data)

    df_ert = pd.DataFrame(all_ert).groupby("base_name").mean(numeric_only=True)
    print(f"ERT DataFrame: {df_ert.shape}")

    ert_datasets = split_ert_by_dimension(df_ert, dims=DIMS)

    for dim in DIMS:
        print(f"\n--- Dimension {dim} ---")
        print(f"  ERT shape: {ert_datasets[dim].shape}")

    solve_rates = compute_solve_rate(ert_datasets)

    for dim in DIMS:
        print(f"\n  Solve rate dim={dim} (top 5):")
        for name, val in solve_rates[dim].tail(5).items():
            print(f"    {val:.3f}  {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analysis pipelines")
    parser.add_argument(
        "--portfolio",
        default="G3PCX_LMCMAES_SPSO",
        help="Portfolio name to analyse CDB impact for (default: G3PCX_LMCMAES_SPSO)",
    )
    args = parser.parse_args()

    run_results_pipeline(portfolio=args.portfolio)
    run_ert_pipeline()
