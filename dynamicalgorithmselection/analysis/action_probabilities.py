from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamicalgorithmselection.analysis.loading import (
    ActionSequence,
    load_action_sequences,
)
from dynamicalgorithmselection.analysis.utils import (
    BBOB_GROUPS,
    FUNCTION_TO_GROUP,
    GROUP_ORDER,
    REPO_ROOT,
    sanitize_filename,
)

DEFAULT_INPUT_PATH = (
    REPO_ROOT / "DAS_CV_G3PCX_LMCMAES_SPSO_PG_CV-LOPO_CDB1.5_DIM10_SEED42_REWARD4.jsonl"
)


def _first_seen_algorithms(sequences: list[ActionSequence]) -> tuple[str, ...]:
    # Sort alphabetically because the JSONL probabilities array
    # aligns with the alphabetical order of the algorithms.
    seen: set[str] = set()
    for sequence in sequences:
        for action in sequence.actions:
            seen.add(action)
    return tuple(sorted(seen))


def _validate_checkpoint_count(sequences: list[ActionSequence]) -> int:
    checkpoint_counts = {len(sequence.actions) for sequence in sequences}
    if len(checkpoint_counts) != 1:
        raise ValueError(
            "All action sequences must have the same number of checkpoints, "
            f"got {sorted(checkpoint_counts)}."
        )
    return checkpoint_counts.pop()


@dataclass
class ActionProbabilityPlotter:
    sequences: list[ActionSequence]
    algorithms: tuple[str, ...]
    checkpoint_count: int

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path) -> "ActionProbabilityPlotter":
        sequences = load_action_sequences(jsonl_path)
        return cls(
            sequences=sequences,
            algorithms=_first_seen_algorithms(sequences),
            checkpoint_count=_validate_checkpoint_count(sequences),
        )

    @property
    def multi_dimension(self) -> bool:
        return len({sequence.dimension for sequence in self.sequences}) > 1

    def _aggregation_label(self, sequence: ActionSequence, mode: str) -> str:
        if mode == "function":
            base_label = sequence.function_id
        elif mode == "group":
            base_label = FUNCTION_TO_GROUP.get(sequence.function_id, "Unknown Group")
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")

        if self.multi_dimension:
            return f"{base_label} (d{sequence.dimension})"
        return base_label

    def _sort_labels(self, labels: list[str], mode: str) -> list[str]:
        if mode == "function":
            return sorted(
                labels,
                key=lambda label: (
                    int(re.search(r"f(\d{3})", label).group(1)),
                    (
                        int(re.search(r"d(\d+)", label).group(1))
                        if "d" in label and "(d" in label
                        else -1
                    ),
                ),
            )

        group_rank = {group_name: rank for rank, group_name in enumerate(GROUP_ORDER)}
        return sorted(
            labels,
            key=lambda label: (
                (
                    int(re.search(r"d(\d+)", label).group(1))
                    if "d" in label and "(d" in label
                    else -1
                ),
                group_rank.get(label.split(" (d", maxsplit=1)[0], math.inf),
            ),
        )

    def aggregate_probabilities(
        self,
        mode: str,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
        """Aggregate per-checkpoint expected action probabilities."""
        grouped_sequences: dict[str, list[ActionSequence]] = defaultdict(list)
        for sequence in self.sequences:
            grouped_sequences[self._aggregation_label(sequence, mode)].append(sequence)

        tables: dict[str, pd.DataFrame] = {}
        sample_sizes: dict[str, int] = {}

        for label in self._sort_labels(list(grouped_sequences), mode):
            sequences = grouped_sequences[label]

            # Shape: (num_algorithms, num_checkpoints)
            counts = np.zeros(
                (len(self.algorithms), self.checkpoint_count), dtype=float
            )

            for sequence in sequences:
                # Add the raw probabilities directly for each checkpoint
                for checkpoint, probs in enumerate(sequence.probabilities):
                    for alg_idx, prob in enumerate(probs):
                        counts[alg_idx, checkpoint] += prob

            # Average the probabilities across all sequences in this group
            mean_probabilities = counts / len(sequences)

            table = pd.DataFrame(
                mean_probabilities,
                index=pd.Index(self.algorithms, name="algorithm"),
                columns=pd.Index(
                    range(1, self.checkpoint_count + 1), name="checkpoint"
                ),
            )
            tables[label] = table
            sample_sizes[label] = len(sequences)

        return tables, sample_sizes

    def save_probability_tables(
        self,
        tables: dict[str, pd.DataFrame],
        output_dir: str | Path,
        prefix: str,
    ) -> list[Path]:
        """Save aggregated probability tables as CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        written_paths: list[Path] = []
        for label, table in tables.items():
            csv_path = output_path / f"{prefix}_{sanitize_filename(label)}.csv"
            table.to_csv(csv_path, float_format="%.6f")
            written_paths.append(csv_path)

        return written_paths

    def save_heatmap_grid(
        self,
        tables: dict[str, pd.DataFrame],
        sample_sizes: dict[str, int],
        output_path: str | Path,
        title: str,
        annotate: bool = False,
    ) -> Path:
        """Save a grid of heatmaps."""
        if not tables:
            raise ValueError("No tables available to plot.")

        n_items = len(tables)
        if n_items <= 4:
            ncols = min(2, n_items)
        elif n_items <= 12:
            ncols = 4
        else:
            ncols = 6
        nrows = math.ceil(n_items / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 3.2, nrows * 2.6),
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes).ravel()

        image = None
        for axis, (label, table) in zip(axes_array, tables.items(), strict=False):
            image = axis.imshow(
                table.to_numpy(),
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(f"{label}\nn={sample_sizes[label]}", fontsize=10)
            axis.set_xticks(range(table.shape[1]), table.columns.tolist())
            axis.set_yticks(range(table.shape[0]), table.index.tolist())
            axis.tick_params(axis="x", labelrotation=45, labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            axis.set_xlabel("Checkpoint")
            axis.set_ylabel("Algorithm")

            if annotate:
                for row_idx in range(table.shape[0]):
                    for col_idx in range(table.shape[1]):
                        value = table.iat[row_idx, col_idx]
                        text_color = "white" if value >= 0.55 else "black"
                        axis.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=text_color,
                        )

        for axis in axes_array[n_items:]:
            axis.set_visible(False)

        if image is None:
            raise RuntimeError("Failed to create heatmaps.")

        colorbar = fig.colorbar(image, ax=axes_array.tolist(), shrink=0.92, pad=0.01)
        colorbar.set_label("Selection probability", rotation=90)
        fig.suptitle(title, fontsize=14)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_report(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate both function-level and group-level plots plus CSV tables."""
        output_dir = Path(output_dir)
        function_tables, function_sizes = self.aggregate_probabilities(mode="function")
        group_tables, group_sizes = self.aggregate_probabilities(mode="group")

        written: dict[str, Path] = {}

        function_table_dir = output_dir / "function_tables"
        group_table_dir = output_dir / "group_tables"
        self.save_probability_tables(function_tables, function_table_dir, "function")
        self.save_probability_tables(group_tables, group_table_dir, "group")

        written["function_heatmaps"] = self.save_heatmap_grid(
            function_tables,
            function_sizes,
            output_dir / "function_probabilities.png",
            title="Expected algorithm probabilities by BBOB function",
            annotate=False,
        )
        written["group_heatmaps"] = self.save_heatmap_grid(
            group_tables,
            group_sizes,
            output_dir / "group_probabilities.png",
            title="Expected algorithm probabilities by standard BBOB group",
            annotate=True,
        )
        return written


def generate_action_probability_report(
    jsonl_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Convenience wrapper for script and library usage."""
    input_path = Path(jsonl_path)
    if output_dir is None:
        output_dir = REPO_ROOT / "analysis_plots" / input_path.stem

    plotter = ActionProbabilityPlotter.from_jsonl(input_path)
    return plotter.generate_report(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate checkpoint-wise expected probabilities and save heatmaps."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL file (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and CSV tables.",
    )
    args = parser.parse_args()

    outputs = generate_action_probability_report(
        jsonl_path=args.input,
        output_dir=args.output_dir,
    )
    resolved_output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "analysis_plots" / args.input.stem
    )

    print(f"Saved probability report to: {resolved_output_dir}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
