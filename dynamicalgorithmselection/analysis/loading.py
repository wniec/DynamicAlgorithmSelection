import json
from dataclasses import dataclass
from pathlib import Path

from dynamicalgorithmselection.analysis.utils import PROBLEM_KEY_RE


@dataclass(frozen=True)
class ActionSequence:
    """One problem instance and its selected algorithms across checkpoints."""

    problem_key: str
    function_id: str
    instance_id: str
    dimension: int
    actions: tuple[str, ...]


def load_action_sequences(jsonl_path: str | Path) -> list[ActionSequence]:
    """Load DAS action sequences from a JSONL file.

    Each line must contain a single ``{problem_key: [algorithm, ...]}`` entry.
    The first value is always 0.0 and should be skipped, this is an artefact.

    Args:
        jsonl_path: Path to the actions JSONL file.

    Returns:
        List of :class:`ActionSequence` objects.
    """
    path = Path(jsonl_path)
    sequences: list[ActionSequence] = []

    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if len(payload) != 1:
                raise ValueError(
                    f"Expected exactly one problem entry on line {line_number}, got {len(payload)}."
                )

            problem_key, actions = next(iter(payload.items()))
            match = PROBLEM_KEY_RE.fullmatch(problem_key)
            if match is None:
                raise ValueError(
                    f"Unsupported problem key format on line {line_number}: {problem_key!r}"
                )
            if not isinstance(actions, list) or not all(
                isinstance(action, str) for action in actions
            ):
                raise ValueError(
                    f"Actions for {problem_key!r} must be a list of strings."
                )

            function_id, instance_id, dimension = match.groups()
            sequences.append(
                ActionSequence(
                    problem_key=problem_key,
                    function_id=function_id,
                    instance_id=instance_id,
                    dimension=int(dimension),
                    actions=tuple(actions),
                )
            )

    if not sequences:
        raise ValueError(f"No action sequences found in {path}.")

    return sequences


def load_experiment_results(
    results_dir: str | Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load JSONL result files from the results directory.

    Each ``.jsonl`` file contains one JSON object per line, keyed by problem
    instance, with metrics like ``area_under_optimization_curve`` and
    ``final_fitness``.

    Args:
        results_dir: Path to the results directory (e.g. ``results_cleaned/results``).

    Returns:
        Nested dict: ``{experiment_name: {problem_key: {metric: value}}}``.
    """
    results_dir = Path(results_dir)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for result_file in sorted(results_dir.iterdir()):
        if result_file.suffix != ".jsonl":
            continue
        experiment_data: dict[str, dict[str, float]] = {}
        with open(result_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                run_results: dict[str, dict[str, float]] = json.loads(line)
                for key, val in run_results.items():
                    experiment_data[key] = val
        results[result_file.stem] = experiment_data

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

    for experiment_file in sorted(ppdata_dir.iterdir()):
        if experiment_file.suffix != ".html" or experiment_file.stem == "index":
            continue
        file_path = experiment_file

        html_content = file_path.read_bytes().decode("iso-8859-1")
        experiment_name = experiment_file.name
        htmls[experiment_name] = html_content

    return htmls
