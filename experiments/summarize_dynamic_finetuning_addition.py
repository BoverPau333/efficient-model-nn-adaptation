"""Genera graficas resumen de adicion para metodos dynamic fine-tuning."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.dynamic_addition_method_comparison import (
    build_dynamic_summaries,
    load_dynamic_addition_rows,
    plot_dynamic_accuracy_vs_time,
    plot_dynamic_dataset_summary,
    plot_dynamic_overall_summary,
    save_dynamic_tables,
    save_dynamic_summaries,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build summary plots for class-addition dynamic fine-tuning methods.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "dynamic_addition_method_comparison"),
        help="Directory where summary tables and plots will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    rows = load_dynamic_addition_rows(results_root)
    if not rows:
        raise FileNotFoundError(f"No dynamic class-addition results found under '{results_root}'.")

    summaries = build_dynamic_summaries(rows)
    save_dynamic_summaries(output_dir, rows, summaries)
    table_paths = save_dynamic_tables(output_dir, summaries)

    overall_plot = plot_dynamic_overall_summary(output_dir, summaries["overall"])
    dataset_plot = plot_dynamic_dataset_summary(output_dir, summaries["by_dataset"])
    landscape_plot = plot_dynamic_accuracy_vs_time(output_dir, summaries["overall"])

    print("Dynamic addition summary saved under:")
    print(f"  {output_dir}")
    print("Tables:")
    for path in table_paths.values():
        print(f"  {path}")
    print("Plots:")
    print(f"  {overall_plot}")
    if dataset_plot is not None:
        print(f"  {dataset_plot}")
    if landscape_plot is not None:
        print(f"  {landscape_plot}")


if __name__ == "__main__":
    main()
