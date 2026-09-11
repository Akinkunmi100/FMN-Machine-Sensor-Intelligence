"""Build the current-state score artifact used by the dashboard at boot.

The shipped model is already refit on all established history. This script
only applies that model to the full sensor frame and persists the result so
the API does not recompute rolling features on every process restart.
"""

import sys

from .config import CURRENT_SCORES_PATH
from .risk_context import score_frame


def main() -> int:
    scored = score_frame(use_cached=False)
    CURRENT_SCORES_PATH.parent.mkdir(exist_ok=True)
    scored.to_parquet(CURRENT_SCORES_PATH, index=False)
    print(f"Saved -> {CURRENT_SCORES_PATH}")
    print(f"{len(scored):,} rows across {scored['machine_id'].nunique()} machines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
