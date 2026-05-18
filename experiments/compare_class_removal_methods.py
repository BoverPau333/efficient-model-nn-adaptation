"""Compara baseline, frozen backbone y fine-tuning para class removal."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.class_removal_method_comparison import (
    filter_completed_rows,
    load_all_method_rows,
    plot_method_comparison,
    save_method_comparison_tables,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and compare class-removal baseline, frozen-backbone and "
            "fine-tuning experiments."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Root results directory containing class_removal_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "class_removal_method_comparison"),
        help="Directory where comparison tables and plots will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    all_rows = load_all_method_rows(results_root)
    if not all_rows:
        raise FileNotFoundError(
            f"No experiments_summary.json files found under '{results_root}' for the configured methods."
        )

    completed_rows = filter_completed_rows(all_rows)
    summaries = save_method_comparison_tables(output_dir, all_rows, completed_rows)
    plot_path = plot_method_comparison(output_dir, summaries["overall"])

    print("Comparison tables saved under:")
    print(f"  {output_dir}")
    print("Main plot saved at:")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
