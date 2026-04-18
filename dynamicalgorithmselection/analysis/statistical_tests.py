"""One-sided Wilcoxon signed-rank tests comparing CDB variants vs CDB=1.0."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dynamicalgorithmselection.analysis.preprocessing import extract_cdb

NON_COMPARED = ["RANDOM", "REWARD"]

_RL_MARKER = "_PG_"
_RANDOM_EXCLUDE = "RANDOM_DAS"


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. NaNs are preserved as NaN."""
    idx = [i for i, p in enumerate(pvals) if not (p is None or np.isnan(p))]
    adjusted = [float("nan")] * len(pvals)
    if not idx:
        return adjusted

    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        val = min(1.0, factor * pvals[i])
        running_max = max(running_max, val)
        adjusted[i] = running_max
    return adjusted


def _aggregate_by_cdb(df: pd.DataFrame, matching: list[str]) -> dict[float, pd.Series]:
    """Collapse duplicate rows sharing the same CDB by averaging per problem."""
    by_cdb: dict[float, list[pd.Series]] = {}
    for name in matching:
        cdb = extract_cdb(name)
        if cdb is None:
            continue
        by_cdb.setdefault(cdb, []).append(df.loc[name].astype(float))
    return {c: pd.concat(rows, axis=1).mean(axis=1) for c, rows in by_cdb.items()}


def _wilcoxon_row(
    treat: pd.Series, base: pd.Series, baseline_cdb: float
) -> dict[str, float]:
    """Run a one-sided (greater) Wilcoxon signed-rank test: treat > base."""
    paired = pd.concat([treat, base], axis=1, keys=["t", "b"]).dropna()
    diffs = (paired["t"] - paired["b"]).to_numpy()
    n = int(len(diffs))
    nonzero = diffs[diffs != 0]

    row = {
        "n_problems": n,
        "n_nonzero": int(len(nonzero)),
        "median_diff": float(np.median(diffs)) if n else float("nan"),
        "mean_diff": float(np.mean(diffs)) if n else float("nan"),
        "statistic": float("nan"),
        "rank_biserial": float("nan"),
        "p_value": float("nan"),
    }

    if len(nonzero) < 1:
        return row

    try:
        res = wilcoxon(
            nonzero,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )
    except ValueError:
        return row

    w_plus = float(res.statistic)
    total = len(nonzero) * (len(nonzero) + 1) / 2.0
    row["statistic"] = w_plus
    row["rank_biserial"] = 2.0 * w_plus / total - 1.0 if total > 0 else float("nan")
    row["p_value"] = float(res.pvalue)
    return row


def compute_cdb_wilcoxon_table(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_path: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
    metric: str = "aocc",
    baseline_cdb: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise one-sided Wilcoxon tests of AOCC(CDB=c) > AOCC(CDB=baseline).

    Runs three families: LOIO standard, LOPO standard, LOPO multi (rows with
    MULTIDIMENSIONAL). Pairs are formed per benchmark problem (columns of
    the AOCC table). Holm adjustment is applied within each (dimension,
    cv_mode, experiment_type) family. Results are written to ``save_path``
    (CSV).
    """
    records: list[dict] = []

    # (cv_mode, experiment_type): filter predicate on row name
    families = {
        ("LOIO", "standard"): lambda n: "MULTIDIMENSIONAL" not in n,
        ("LOPO", "standard"): lambda n: "MULTIDIMENSIONAL" not in n,
        ("LOPO", "multi"): lambda n: "MULTIDIMENSIONAL" in n,
    }

    for (cv_mode, exp_type), predicate in families.items():
        for dim in dims:
            key = f"{metric}_{cv_mode}"
            df = datasets.get(dim, {}).get(key)
            if df is None or df.empty:
                continue

            matching = [
                n
                for n in df.index
                if any(("_".join(i) in n) for i in permutations(portfolio))
                and predicate(n)
                and all(nc not in n for nc in NON_COMPARED)
            ]
            if not matching:
                continue

            by_cdb = _aggregate_by_cdb(df, matching)
            if baseline_cdb not in by_cdb:
                continue
            base = by_cdb[baseline_cdb]

            family_rows: list[dict] = []
            for cdb, treat in by_cdb.items():
                if cdb == baseline_cdb:
                    continue
                rec = _wilcoxon_row(treat, base, baseline_cdb)
                rec.update(
                    {
                        "dim": dim,
                        "cv_mode": cv_mode,
                        "experiment_type": exp_type,
                        "metric": metric.upper(),
                        "cdb": cdb,
                        "baseline_cdb": baseline_cdb,
                    }
                )
                family_rows.append(rec)

            adj = _holm([r["p_value"] for r in family_rows])
            for r, p_adj in zip(family_rows, adj):
                r["p_value_holm"] = p_adj
                r["significant"] = bool(p_adj < alpha) if not np.isnan(p_adj) else False
            records.extend(family_rows)

    columns = [
        "dim",
        "cv_mode",
        "experiment_type",
        "metric",
        "cdb",
        "baseline_cdb",
        "n_problems",
        "n_nonzero",
        "median_diff",
        "mean_diff",
        "statistic",
        "rank_biserial",
        "p_value",
        "p_value_holm",
        "significant",
    ]
    table = (
        pd.DataFrame(records, columns=columns)
        .sort_values(["cv_mode", "experiment_type", "dim", "cdb"], kind="stable")
        .reset_index(drop=True)
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(save_path, index=False, float_format="%.6g")
    return table


def compute_rl_vs_random_wilcoxon_table(
    datasets: dict[int, dict[str, pd.DataFrame]],
    portfolio: list[str],
    save_path: Path,
    dims: tuple[int, ...] = (2, 3, 5, 10),
    metric: str = "aocc",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One-sided Wilcoxon tests of RL-Exponential-DAS > Exponential-Random.

    For each (cv_mode, dim, cdb_value) pair, tests whether the policy-gradient
    agent (RL-Exponential-DAS) significantly outperforms the random baseline
    (Exponential-Random) at the same CDB value.  Three families are tested:
    LOIO standard, LOPO standard, and LOPO multi (MULTIDIMENSIONAL rows).
    Holm-Bonferroni correction is applied within each (cv_mode, exp_type)
    family across dimensions and CDB values.

    Args:
        datasets: Per-dimension metric DataFrames produced by
            ``split_results_by_dimension``.
        portfolio: Algorithm names used to identify matching experiments.
        save_path: CSV output path for the results table.
        dims: Dimensions to test.
        metric: Metric key prefix (default ``"aocc"``).
        alpha: Significance threshold after Holm correction.

    Returns:
        DataFrame with one row per (dim, cv_mode, exp_type, cdb) comparison.
    """
    records: list[dict] = []

    families = {
        ("LOIO", "standard"): lambda n: "MULTIDIMENSIONAL" not in n,
        ("LOPO", "standard"): lambda n: "MULTIDIMENSIONAL" not in n,
        ("LOPO", "multi"): lambda n: "MULTIDIMENSIONAL" in n,
    }

    for (cv_mode, exp_type), predicate in families.items():
        family_rows: list[dict] = []

        for dim in dims:
            key = f"{metric}_{cv_mode}"
            df = datasets.get(dim, {}).get(key)
            if df is None or df.empty:
                continue

            rl_rows = [
                n
                for n in df.index
                if any("_".join(i) in n for i in permutations(portfolio))
                and _RL_MARKER in n
                and predicate(n)
                and extract_cdb(n) is not None
                and extract_cdb(n) != 1.0
            ]

            random_rows = [
                n
                for n in df.index
                if "RANDOM" in n
                and _RANDOM_EXCLUDE not in n
                and extract_cdb(n) is not None
            ]

            if not rl_rows or not random_rows:
                continue

            rl_by_cdb = _aggregate_by_cdb(df, rl_rows)
            random_by_cdb = _aggregate_by_cdb(df, random_rows)

            common_cdbs = sorted(set(rl_by_cdb) & set(random_by_cdb))
            for cdb in common_cdbs:
                rec = _wilcoxon_row(rl_by_cdb[cdb], random_by_cdb[cdb], cdb)
                rec.update(
                    {
                        "dim": dim,
                        "cv_mode": cv_mode,
                        "experiment_type": exp_type,
                        "metric": metric.upper(),
                        "cdb": cdb,
                        "treatment": "RL-Exponential-DAS",
                        "baseline": "Exponential-Random",
                    }
                )
                family_rows.append(rec)

        adj = _holm([r["p_value"] for r in family_rows])
        for r, p_adj in zip(family_rows, adj):
            r["p_value_holm"] = p_adj
            r["significant"] = bool(p_adj < alpha) if not np.isnan(p_adj) else False
        records.extend(family_rows)

    columns = [
        "dim",
        "cv_mode",
        "experiment_type",
        "metric",
        "cdb",
        "treatment",
        "baseline",
        "n_problems",
        "n_nonzero",
        "median_diff",
        "mean_diff",
        "statistic",
        "rank_biserial",
        "p_value",
        "p_value_holm",
        "significant",
    ]
    table = (
        pd.DataFrame(records, columns=columns)
        .sort_values(["cv_mode", "experiment_type", "dim", "cdb"], kind="stable")
        .reset_index(drop=True)
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(save_path, index=False, float_format="%.6g")
    return table
