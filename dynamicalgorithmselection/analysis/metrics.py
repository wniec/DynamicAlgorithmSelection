"""Metric extraction and computation."""

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from autorank import autorank
from bs4 import BeautifulSoup, Tag


def extract_metric(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> pd.DataFrame:
    """Extract a single metric from loaded experiment results into a DataFrame.

    Args:
        results: Nested dict as returned by
            `load_experiment_results` (experiment -> problem -> metric -> value).
        metric: Metric key to extract (e.g., "area_under_optimization_curve"
            or "final_fitness").

    Returns:
        DataFrame with experiments as rows and problems as columns.
    """
    data: dict[str, dict[str, float]] = defaultdict(dict)

    for experiment, problems in results.items():
        for problem_key, metrics in problems.items():
            if metric in metrics:
                data[experiment][problem_key] = metrics[metric]
    return pd.DataFrame.from_dict(data, orient="index")


def parse_ert_from_html(
    html_content: str,
    targets: tuple[str, ...] = ("1e-5", "1e-7"),
) -> dict[str, float]:
    """Parse ERT ratios from a COCO `pptable.html` file.

    Extracts the ERT (Expected Runtime) divided by the best BBOB-2009
    algorithm's ERT for specified precision targets.

    Args:
        html_content: Raw HTML string from `pptable.html`.
        targets: Column headers (precision targets) to extract.

    Returns:
        Dict mapping "DIM{d}_f{nn}_target_{t}" to the ERT ratio value.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    extracted: dict[str, float] = {}

    for table in soup.find_all("table"):
        dimension = _parse_dimension(table)
        if dimension is None:
            continue

        headers = _extract_table_headers(table)
        if not headers:
            continue

        extracted.update(_parse_table_rows(table, headers, dimension, targets))

    return extracted


def _extract_table_headers(table: Tag) -> list[str]:
    """Extract column headers from an HTML table's thead."""
    thead = table.find("thead")
    if not thead:
        return []
    return [td.get_text(strip=True) for td in thead.find_all(["th", "td"])][1:]


def _parse_table_rows(
    table: Tag, headers: list[str], dimension: int, targets: tuple[str, ...]
) -> dict[str, float]:
    """Parse the tbody of an HTML table for function ERT ratios."""
    tbody = table.find("tbody")
    if not tbody:
        return {}

    extracted = {}
    current_func: str | None = None

    for row in tbody.find_all("tr"):
        th = row.find("th")
        tds = [td.get_text(strip=True) for td in row.find_all("td")]
        th_text = th.get_text(strip=True) if th else ""

        # A populated `th` indicates the start of a new function block
        if th_text:
            current_func = th_text

        # An empty `th` while inside a function block indicates the ratio row
        elif current_func and not th_text:
            func_num = int(current_func[1:])
            for i, header in enumerate(headers):
                if header in targets and i < len(tds):
                    raw_value = tds[i].split(" ")[0]
                    ratio = float("inf") if "∞" in raw_value else float(raw_value)
                    key = f"DIM{dimension}_f{func_num:02d}_target_{header}"
                    extracted[key] = ratio

    return extracted


def compute_solve_rate(
    datasets: dict[int, pd.DataFrame],
) -> dict[int, pd.Series]:
    """Compute the fraction of problems solved (non-infinite ERT) per experiment.

    Args:
        datasets: Per-dimension ERT DataFrames.

    Returns:
        Dictionary mapping dimension to a Series of solve rates, sorted ascending.
    """
    solve_rates: dict[int, pd.Series] = {}
    for dim, df in datasets.items():
        is_inf = np.isinf(df).astype(float)
        rate = (1 - is_inf.mean(axis=1)).sort_values()
        solve_rates[dim] = rate
    return solve_rates


def compute_ERT_rank(
    datasets: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """Compute the mean rank of Expected Runtimes (ERT) per experiment.

    Because lower ERTs are better, this function uses `order='ascending'`.
    Infinite values (failed runs) are penalized to the worst ranks.

    Args:
        datasets: Per-dimension ERT DataFrames.

    Returns:
        Dictionary mapping dimension to a Series of mean ERT ranks.
    """
    solve_rates: dict[int, pd.DataFrame] = {}
    for dim, df in datasets.items():
        rankdf = autorank(df.T, alpha=0.05, verbose=False, order="ascending").rankdf
        rate = pd.DataFrame(rankdf[["meanrank"]])
        solve_rates[dim] = rate
    return solve_rates


def _parse_dimension(table: Tag) -> int | None:
    """Extract the dimension number from the paragraph preceding a table."""
    prev_p = table.find_previous_sibling("p")
    if prev_p:
        dim_match = re.search(r"Dimension\s*=\s*(\d+)", prev_p.text)
        if dim_match:
            return int(dim_match.group(1))
    return None
