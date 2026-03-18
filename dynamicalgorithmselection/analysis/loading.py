import json
from pathlib import Path


def load_experiment_results(
    results_dir: str | Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load JSON result files from the results directory.

    Each experiment subdirectory contains per-problem JSON files with metrics
    like ``area_under_optimization_curve`` and ``final_fitness``.

    Args:
        results_dir: Path to the results directory (e.g. ``results_cleaned/results``).

    Returns:
        Nested dict: ``{experiment_name: {problem_key: {metric: value}}}``.
    """
    results_dir = Path(results_dir)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for experiment in sorted(results_dir.iterdir()):
        if not experiment.is_dir():
            continue
        experiment_data: dict[str, dict[str, float]] = {}
        for result_file in sorted(experiment.iterdir()):
            if not result_file.suffix == ".json":
                continue
            with open(result_file) as f:
                run_results: dict[str, dict[str, float]] = json.load(f)
            for key, val in run_results.items():
                experiment_data[key] = val
        results[experiment.name] = experiment_data

    return results


def load_ert_htmls(
    ppdata_dir: str | Path,
) -> dict[str, str]:
    """Load ``pptable.html`` files from COCO post-processing output.

    Reads each experiment's ``pptable.html`` using ISO-8859-1 encoding and
    strips the timestamp suffix from directory names to produce clean
    experiment keys.

    Args:
        ppdata_dir: Path to the ppdata directory (e.g. ``results_cleaned/ppdata``).

    Returns:
        Dict mapping experiment base name to HTML content string.
    """
    ppdata_dir = Path(ppdata_dir)
    htmls: dict[str, str] = {}

    for experiment_dir in sorted(ppdata_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        pptable_path = experiment_dir / "pptable.html"
        if not pptable_path.exists():
            continue

        html_content = pptable_path.read_bytes().decode("iso-8859-1")
        # Strip timestamp suffix (e.g. "_031107h3301") from directory name
        experiment_name = "_".join(experiment_dir.name.split("_")[:-1])
        htmls[experiment_name] = html_content

    return htmls
