"""CLI script to validate the movie dataset and output a structured Data Quality Report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.config import AppConfig  # noqa: E402
from cinematch.data.loader import MovieDataLoader  # noqa: E402
from cinematch.utils.exceptions import CineMatchError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CineMatch movie metadata dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to movies_metadata.csv (defaults to AppConfig dataset_path).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional reproducible sample size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path

    try:
        loader = MovieDataLoader(
            dataset_path=dataset_path,
            sample_size=args.sample_size,
            random_state=args.random_state,
        )
        loaded = loader.load()
    except CineMatchError as exc:
        print(json.dumps({"status": "error", "error_code": exc.error_code, "message": exc.message}, indent=2))
        return 1

    output = {
        "status": "valid",
        "dataset_path": str(dataset_path),
        "quality_report": loaded.quality_report.to_dict(),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
