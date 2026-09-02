from __future__ import annotations

import argparse
import json

from backend.forecast_logic import write_forecast_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a forecast artifact for a target date.")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format. Defaults to latest predictable date.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/forecasts",
        help="Directory where forecast csv/json files will be written.",
    )
    args = parser.parse_args()

    result = write_forecast_files(target_date=args.target_date, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
