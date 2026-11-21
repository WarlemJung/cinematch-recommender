"""CLI script to download and prepare optional MovieLens user ratings mapped to CineMatch TMDB IDs."""

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
from cinematch.data.movielens import prepare_movielens_ratings  # noqa: E402
from cinematch.utils.exceptions import CineMatchError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare MovieLens ml-latest-small ratings mapped to CineMatch catalog IDs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to movies_metadata.csv (defaults to AppConfig dataset_path).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ratings.csv",
        help="Output CSV path for prepared ratings (default: data/ratings.csv).",
    )
    parser.add_argument(
        "--source-zip",
        type=str,
        default=None,
        help="Optional local ml-latest-small.zip path to avoid network download.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Optional local extracted MovieLens directory containing ratings.csv and links.csv.",
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path

    try:
        report = prepare_movielens_ratings(
            movies_metadata_path=dataset_path,
            output_ratings_path=args.output,
            source_zip=args.source_zip,
            source_dir=args.source_dir,
            download_if_missing=True,
        )
    except CineMatchError as exc:
        print(json.dumps({"status": "error", "error_code": exc.error_code, "message": exc.message}, indent=2))
        return 1

    print(json.dumps({"status": "success", "report": report.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
