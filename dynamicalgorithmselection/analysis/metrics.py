"""Metric extraction and computation."""

import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def extract_metric(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> pd.DataFrame:
    """Extract a single metric from loaded experiment results into a DataFrame.

    Args:
        results: Nested dict as returned by
            :func:`~dynamicalgorithmselection.analysis.loading.load_experiment_results`.
        metric: Metric key to extract (e.g. ``"area_under_optimization_curve"``
            or ``"final_fitness"``).

    Returns:
        DataFrame with experiments as rows and problems as columns.
    """
    data: dict[str, dict[str, float]] = {}
    for experiment, problems in results.items():
        data[experiment] = {}
        for problem_key, metrics in problems.items():
            if metric in metrics:
                data[experiment][problem_key] = metrics[metric]
    return pd.DataFrame.from_dict(data, orient="index")


def parse_ert_from_html(
    html_content: str,
    targets: tuple[str, ...] = ("1e-5", "1e-7"),
) -> dict[str, float]:
    """Parse ERT ratios from a COCO ``pptable.html`` file.

    Extracts the ERT (Expected Runtime) divided by the best BBOB-2009
    algorithm's ERT for specified precision targets. Only the second row
    per function block (the ratio row) is extracted.

    Args:
        html_content: Raw HTML string from ``pptable.html``.
        targets: Column headers (precision targets) to extract.

    Returns:
        Dict mapping ``"DIM{d}_f{nn}_target_{t}"`` to the ERT ratio value.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    extracted: dict[str, float] = {}

    for table in soup.find_all("table"):
        dimension = _parse_dimension(table)
        if dimension is None:
            continue

        thead = table.find("thead")
        if not thead:
            continue
        headers = [td.get_text(strip=True) for td in thead.find_all(["th", "td"])][1:]

        tbody = table.find("tbody")
        if not tbody:
            continue

        current_func: str | None = None
        for row in tbody.find_all("tr"):
            th = row.find("th")
            tds = [td.get_text(strip=True) for td in row.find_all("td")]

            if th and th.get_text(strip=True):
                current_func = th.get_text(strip=True)
            elif current_func and (not th or not th.get_text(strip=True)):
                func_num = int(current_func[1:])
                for i, header in enumerate(headers):
                    if header in targets and i < len(tds):
                        raw_value = tds[i].split(" ")[0]
                        if "∞" in raw_value:
                            ratio = float("inf")
                        else:
                            ratio = float(raw_value)
                        key = f"DIM{dimension}_f{func_num:02d}_target_{header}"
                        extracted[key] = ratio

    return extracted


def compute_solve_rate(
    datasets: dict[int, pd.DataFrame],
) -> dict[int, pd.Series]:
    """Compute the fraction of problems solved (non-infinite ERT) per experiment.

    Args:
        datasets: Per-dimension ERT DataFrames as returned by
            :func:`~dynamicalgorithmselection.analysis.preprocessing.split_ert_by_dimension`.

    Returns:
        ``{dim: Series}`` with solve rates sorted ascending.
    """
    solve_rates: dict[int, pd.Series] = {}
    for dim, df in datasets.items():
        is_inf = np.isinf(df).astype(float)
        rate = (1 - is_inf.mean(axis=1)).sort_values()
        solve_rates[dim] = rate
    return solve_rates


def _parse_dimension(table) -> int | None:  # noqa: ANN001
    """Extract the dimension number from the paragraph preceding a table."""
    prev_p = table.find_previous_sibling("p")
    if prev_p:
        dim_match = re.search(r"Dimension\s*=\s*(\d+)", prev_p.text)
        if dim_match:
            return int(dim_match.group(1))
    return None
