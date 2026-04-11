from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches keys like "bbob_f006_i01_d10" and captures (function_id, instance_id, dimension).
PROBLEM_KEY_RE = re.compile(r"^bbob_(f\d{3})_(i\d{2})_d(\d+)$")

# Standard BBOB groups used in the 24-function noiseless suite.
BBOB_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Separable", tuple(f"f{i:03d}" for i in range(1, 6))),
    ("Low/Moderate Conditioning", tuple(f"f{i:03d}" for i in range(6, 10))),
    ("High Conditioning, Unimodal", tuple(f"f{i:03d}" for i in range(10, 15))),
    (
        "Multi-modal, Adequate Structure",
        tuple(f"f{i:03d}" for i in range(15, 20)),
    ),
    ("Multi-modal, Weak Structure", tuple(f"f{i:03d}" for i in range(20, 25))),
)
FUNCTION_TO_GROUP = {
    function_id: group_name
    for group_name, function_ids in BBOB_GROUPS
    for function_id in function_ids
}
GROUP_ORDER = [group_name for group_name, _ in BBOB_GROUPS]


def sanitize_filename(name: str) -> str:
    """Convert a label to a safe filename-compatible string."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
