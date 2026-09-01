from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_PATHS = (
    ROOT / "fetch_fifa_official_rankings.py",
    ROOT / "fetch_transfermarkt_world_cup_values.py",
    ROOT / "data" / "fifa_rankings_annual_start.csv",
    ROOT / "data" / "fifa_rankings_history_datofutbol.csv",
    ROOT / "data" / "fifa_rankings_official_snapshots.csv",
    ROOT / "data" / "transfermarkt_world_cup_2026_values.csv",
    ROOT / "data" / "world_cup_2026_key_player_signals.csv",
)
EXPECTED_ERROR = "missing required FIFA ranking input"


def main() -> int:
    present = [str(path.relative_to(ROOT)) for path in PROHIBITED_PATHS if path.exists()]
    if present:
        raise RuntimeError(f"restricted files are present in the public checkout: {present}")

    result = subprocess.run(
        [sys.executable, "predict_fifa_profile.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode == 0:
        raise RuntimeError("full prediction unexpectedly succeeded without restricted inputs")
    if EXPECTED_ERROR not in output:
        raise RuntimeError(
            "full prediction failed for an unexpected reason:\n"
            + "\n".join(output.strip().splitlines()[-12:])
        )

    print("Restricted inputs are absent and the model fails with the documented error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
