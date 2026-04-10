import re

import pandas as pd


def extract_seed_info(
    df: pd.DataFrame,
    max_seed: int = 99,
) -> pd.DataFrame:
    """Add ``base_name`` column and filter to valid seeds.

    Parses the ``_SEED<N>`` suffix from the DataFrame index to separate the
    base experiment configuration from the seed number. Rows with seeds
    outside ``[0, max_seed]`` are dropped; unseeded experiments are kept.

    Args:
        df: DataFrame indexed by experiment name.
        max_seed: Maximum seed value to keep (inclusive).

    Returns:
        DataFrame with ``base_name`` column added and invalid seeds removed.
    """
    base_names: list[str] = []
    seeds: list[int | None] = []

    for exp_name in df.index:
        seed_match = re.search(r"_SEED(\d+)", str(exp_name))
        if seed_match:
            base_names.append(str(exp_name)[: seed_match.start()])
            seeds.append(int(seed_match.group(1)))
        else:
            base_names.append(str(exp_name))
            seeds.append(None)

    result = df.copy()
    result["base_name"] = base_names
    result["seed"] = seeds

    valid_mask = result["seed"].isna() | result["seed"].isin(range(max_seed + 1))
    return result[valid_mask].drop(columns=["seed"])


def aggregate_over_seeds(
    df: pd.DataFrame,
    max_seed: int = 9999,
) -> pd.DataFrame:
    """Group experiments by base name (ignoring seed suffix) and average.

    Args:
        df: DataFrame indexed by experiment name.
        max_seed: Maximum seed value to keep (inclusive).

    Returns:
        DataFrame indexed by base experiment name with mean values.
    """
    with_base = extract_seed_info(df, max_seed=max_seed)
    return with_base.groupby("base_name").mean(numeric_only=True)


def split_results_by_dimension(
    auoc: pd.DataFrame,
    final_fitness: pd.DataFrame,
    aocc: pd.DataFrame,
    dims: tuple[int, ...] = (2, 3, 5, 10),
    extra_baselines: list[str] | None = None,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Split AUOC and final_fitness DataFrames by dimension and CV mode.

    Produces per-dimension datasets filtered by LOIO/LOPO cross-validation
    mode. Multidimensional experiments and baselines are included in each
    dimension's dataset with their columns filtered to match.

    Args:
        auoc: Aggregated AUOC DataFrame (experiments x problems).
        final_fitness: Aggregated final fitness DataFrame (experiments x problems).
        dims: Tuple of dimensions to split by.
        extra_baselines: Additional baseline experiment names to include
            (e.g. ``["BASELINES_baselines_MADDE"]``).

    Returns:
        ``{dim: {"auoc_LOIO": df, "auoc_LOPO": df,
                 "final_fitness_LOIO": df, "final_fitness_LOPO": df}}``.
    """
    all_rows = list(auoc.index)

    multidim_rows = [r for r in all_rows if "MULTIDIMENSIONAL" in r]
    baseline_rows = [r for r in all_rows if "RANDOM" in r and "RANDOM_DAS" not in r]
    if extra_baselines:
        for name in extra_baselines:
            if name in auoc.index and name not in baseline_rows:
                baseline_rows.append(name)

    extreme_rows = [r for r in all_rows if "best" in r or "worst" in r]
    global_rows = list(set(baseline_rows + multidim_rows + extreme_rows))

    # Build global datasets (filtered to only rows that exist)
    existing_global = [r for r in global_rows if r in auoc.index]
    global_auoc = auoc.loc[existing_global]
    global_ff = final_fitness.loc[existing_global]
    global_aocc = auoc.loc[existing_global]

    datasets: dict[int, dict[str, pd.DataFrame]] = {}

    for dim in dims:
        dim_rows = [
            r
            for r in all_rows
            if f"_DIM{dim}" in r and r not in multidim_rows and r not in extreme_rows
        ]

        columns = [c for c in auoc.columns if int(c[-2:]) == dim]

        # Select multidim/baseline data for this dimension's columns
        md_auoc = global_auoc[columns].copy()
        md_ff = global_ff[columns].copy()
        md_aocc = global_aocc[columns].copy()

        md_auoc.index = md_auoc.index.map(lambda x: f"{x}_{dim}")
        md_ff.index = md_ff.index.map(lambda x: f"{x}_{dim}")
        md_aocc.index = md_aocc.index.map(lambda x: f"{x}_{dim}")

        auoc_combined = pd.concat([auoc.loc[dim_rows].dropna(axis=1), md_auoc])
        ff_combined = pd.concat(
            [final_fitness.loc[dim_rows].dropna(axis=1), md_ff]
        )

        aocc_combined = pd.concat([aocc.loc[dim_rows].dropna(axis=1), md_aocc])
        base_data = {
            "auoc": auoc_combined,
            "final_fitness": ff_combined,
            "aocc": aocc_combined,
        }

        # LOIO keeps rows without "LOPO", LOPO keeps rows without "LOIO"
        cv_filters = {"LOIO": "LOPO", "LOPO": "LOIO"}

        datasets[dim] = {}
        for metric_name, df in base_data.items():
            for suffix, exclude_str in cv_filters.items():
                key = f"{metric_name}_{suffix}"
                datasets[dim][key] = df[~df.index.str.contains(exclude_str, na=False)]

    return datasets


def extract_cdb(experiment_name: str) -> float | None:
    """Extract CDB value from an experiment name.

    Handles both ``CDB1.5`` and bare ``_1.5_`` formats.
    """
    m = re.search(r"CDB([\d.]+)", experiment_name)
    if m:
        return float(m.group(1))
    m = re.search(r"_CV-(?:LOIO|LOPO)_([\d.]+)_DIM", experiment_name)
    if m:
        return float(m.group(1))
    return None


def split_ert_by_dimension(
    df: pd.DataFrame,
    dims: tuple[int, ...] = (2, 3, 5, 10),
) -> dict[int, pd.DataFrame]:
    """Split an ERT DataFrame by dimension.

    Identifies dimension-specific experiments, multidimensional experiments,
    random baselines, and baseline experiments, then combines them per
    dimension.

    Args:
        df: Aggregated ERT DataFrame (experiments x targets).
        dims: Tuple of dimensions to split by.

    Returns:
        ``{dim: DataFrame}`` with duplicated indices removed.
    """
    all_experiments = df.index.tolist()

    multidim_rows = [e for e in all_experiments if "MULTIDIMENSIONAL" in e]
    global_random_rows = [
        e for e in all_experiments if "RANDOM" in e and "_DIM" not in e
    ]
    baseline_rows = [e for e in all_experiments if "BASELINES" in e]

    global_runs = list(set(multidim_rows + global_random_rows + baseline_rows))
    existing_global = [r for r in global_runs if r in df.index]
    global_dataset = df.loc[existing_global]

    datasets: dict[int, pd.DataFrame] = {}

    for dim in dims:
        dim_marker = f"_DIM{dim}"
        dim_rows = [e for e in all_experiments if dim_marker in e]
        columns = [c for c in df.columns if c.startswith(f"DIM{dim}_")]

        dim_specific = df.loc[dim_rows, columns]
        global_selected = global_dataset[columns]

        combined = pd.concat([dim_specific, global_selected])
        combined = combined[~combined.index.duplicated(keep="first")]
        datasets[dim] = combined.dropna(axis=0, how="all").dropna(axis=1, how="all")

    return datasets
