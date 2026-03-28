import os
import json

import cocoex
import numpy as np
from tqdm import tqdm


def generate(
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
    output_file: str = "global_optima.jsonl",
):
    """
    Extracts global optima and writes them to a JSONL file.
    """
    # Open the file once outside the loop for efficiency
    with open(output_file, "w", encoding="utf-8") as f:
        for problem_id in tqdm(problem_ids, smoothing=0.0):
            problem_instance = problems_suite.get_problem(problem_id)
            optimum = get_global_minimum(problem_instance)

            # Convert numpy types to native Python types for JSON serialization
            if isinstance(optimum, (list, np.ndarray)):
                opt_val = [float(val) for val in optimum]
            else:
                opt_val = float(optimum)

            # Create the dictionary entry
            record = {problem_id: opt_val}

            # Write the JSON line and add a newline character
            f.write(json.dumps(record) + "\n")

            problem_instance.free()


def get_global_minimum(problem):
    """
    Extracts the true global minimum (f_opt) from a COCO problem
    using the file-dump workaround, leaving no trace behind.
    """
    # The default hardcoded file COCO writes to
    file_name = "._bbob_problem_best_parameter.txt"

    # 1. Trigger the C-backend to write the optimum coordinates to disk
    problem._best_parameter("print")

    # 2. Read the coordinates and immediately delete the file
    if os.path.exists(file_name):
        x_opt = np.loadtxt(file_name)
        os.remove(file_name)
    else:
        raise FileNotFoundError("COCO failed to write the best parameter file.")

    # 3. Evaluate the coordinates to get the exact global minimum
    return problem(x_opt)


if __name__ == "__main__":
    suite = cocoex.Suite("bbob", "", "")
    sample_ids = [i.id for i in suite]

    generate(suite, sample_ids, output_file="bbob_optima.jsonl")
    print("Generation complete! Check bbob_optima.jsonl")
