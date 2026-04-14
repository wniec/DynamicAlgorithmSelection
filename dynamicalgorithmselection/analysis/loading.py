import json
import re
from dataclasses import dataclass
from pathlib import Path

from dynamicalgorithmselection.analysis.utils import PROBLEM_KEY_RE

#: Regex to parse relevant behaviour filenames, e.g.
#: DAS_CV_G3PCX_LMCMAES_SPSO_PG_CV-LOIO_CDB1.0_DIM2_SEED12.jsonl
_BEHAVIOUR_FILE_RE = re.compile(
    r"DAS_CV_(.+)_PG_CV-(\w+)_(CDB[\d.]+)_DIM(\d+)_SEED(\d+)\.jsonl$"
)


@dataclass(frozen=True)
class BehaviourFile:
    """Parsed metadata from a behaviour JSONL filename."""

    path: Path
    portfolio: str  # e.g. "G3PCX_LMCMAES_SPSO"
    exp_type: str  # "LOIO" or "LOPO"
    cdb: str  # e.g. "CDB1.0"
    dim: int  # 2, 3, 5, or 10
    seed: int  # 12, 23, or 34


def discover_behaviour_files(
    behaviour_dir: Path,
    portfolio: str = "G3PCX_LMCMAES_SPSO",
    exp_types: tuple[str, ...] = ("LOIO", "LOPO"),
) -> list[BehaviourFile]:
    """Discover and parse all matching behaviour files."""
    files: list[BehaviourFile] = []
    for path in sorted(behaviour_dir.glob("*.jsonl")):
        m = _BEHAVIOUR_FILE_RE.match(path.name)
        if m is None:
            continue
        p, et, cdb, dim, seed = m.groups()
        if p != portfolio or et not in exp_types:
            continue
        files.append(
            BehaviourFile(
                path=path,
                portfolio=p,
                exp_type=et,
                cdb=cdb,
                dim=int(dim),
                seed=int(seed),
            )
        )
    return files


@dataclass(frozen=True)
class ActionSequence:
    """One problem instance and its selected algorithms across checkpoints."""

    problem_key: str
    function_id: str
    instance_id: str
    dimension: int
    actions: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]


def load_action_sequences(jsonl_path: str | Path) -> list[ActionSequence]:
    """Load DAS action sequences from a JSONL file.

    Each line must contain a single ``{problem_key: [actions_list, probabilities_list]}`` entry.

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

            problem_key, data = next(iter(payload.items()))

            # Validate the new schema: value should be a list of exactly 2 items
            if not isinstance(data, list) or len(data) != 2:
                raise ValueError(
                    f"Expected a list of [actions, probabilities] for {problem_key!r} on line {line_number}."
                )

            actions, probabilities = data

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

            # Validate probabilities structure
            if not isinstance(probabilities, list) or not all(
                isinstance(prob_list, list) for prob_list in probabilities
            ):
                raise ValueError(
                    f"Probabilities for {problem_key!r} must be a list of lists of floats."
                )

            function_id, instance_id, dimension = match.groups()
            sequences.append(
                ActionSequence(
                    problem_key=problem_key,
                    function_id=function_id,
                    instance_id=instance_id,
                    dimension=int(dimension),
                    actions=tuple(actions),
                    # Convert list of lists to tuple of tuples for the frozen dataclass
                    probabilities=tuple(tuple(p) for p in probabilities),
                )
            )

    if not sequences:
        raise ValueError(f"No action sequences found in {path}.")

    return sequences


@dataclass(frozen=True)
class DiversitySequence:
    """One problem instance and its population diversity values across checkpoints.

    The first diversity value in the raw file (always 0.0) is an artefact and
    is skipped during loading, so ``diversity[0]`` corresponds to checkpoint 1.
    """

    problem_key: str
    function_id: str
    instance_id: str
    dimension: int
    diversity: tuple[float, ...]


def load_diversity_sequences(jsonl_path: str | Path) -> list[DiversitySequence]:
    """Load population diversity sequences from a JSONL file.

    Each line must contain a single ``{problem_key: [float, ...]}`` entry.
    The first value is always 0.0 (artefact) and is skipped automatically.

    Args:
        jsonl_path: Path to the diversity JSONL file.

    Returns:
        List of :class:`DiversitySequence` objects (first value already dropped).
    """
    path = Path(jsonl_path)
    sequences: list[DiversitySequence] = []

    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if len(payload) != 1:
                raise ValueError(
                    f"Expected exactly one entry on line {line_number}, got {len(payload)}."
                )

            problem_key, diversity_values = next(iter(payload.items()))
            match = PROBLEM_KEY_RE.fullmatch(problem_key)
            if match is None:
                raise ValueError(
                    f"Unsupported problem key on line {line_number}: {problem_key!r}"
                )
            if not isinstance(diversity_values, list) or len(diversity_values) < 2:
                raise ValueError(
                    f"Expected a list with at least 2 values for {problem_key!r}."
                )

            function_id, instance_id, dimension = match.groups()
            sequences.append(
                DiversitySequence(
                    problem_key=problem_key,
                    function_id=function_id,
                    instance_id=instance_id,
                    dimension=int(dimension),
                    diversity=tuple(float(v) for v in diversity_values[1:]),
                )
            )

    if not sequences:
        raise ValueError(f"No diversity sequences found in {path}.")

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
        experiment_name = result_file.stem
        experiment_name = re.sub(r"_(\d\.\d)", r"_RANDOM_CDB\1", experiment_name)

        results[experiment_name] = experiment_data

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
