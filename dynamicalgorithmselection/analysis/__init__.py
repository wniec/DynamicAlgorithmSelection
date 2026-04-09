from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    load_action_sequences,
    load_experiment_results,
    load_ert_htmls,
)
from dynamicalgorithmselection.analysis.preprocessing import (
    aggregate_over_seeds,
    extract_seed_info,
    split_results_by_dimension,
    split_ert_by_dimension,
)
from dynamicalgorithmselection.analysis.metrics import (
    extract_metric,
    parse_ert_from_html,
    compute_solve_rate,
)

__all__ = [
    "load_experiment_results",
    "load_ert_htmls",
    "load_action_sequences",
    "ActionSequence",
    "ActionProbabilityPlotter",
    "generate_action_probability_report",
    "DiversitySequence",
    "DiversityAnalyser",
    "load_diversity_sequences",
    "generate_diversity_report",
    "aggregate_over_seeds",
    "extract_seed_info",
    "split_results_by_dimension",
    "split_ert_by_dimension",
    "extract_metric",
    "parse_ert_from_html",
    "compute_solve_rate",
]


def __getattr__(name: str):
    if name in {"ActionProbabilityPlotter", "generate_action_probability_report"}:
        from dynamicalgorithmselection.analysis import action_probabilities

        return getattr(action_probabilities, name)
    if name in {
        "DiversitySequence",
        "DiversityAnalyser",
        "load_diversity_sequences",
        "generate_diversity_report",
    }:
        from dynamicalgorithmselection.analysis import diversity_analysis

        return getattr(diversity_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
