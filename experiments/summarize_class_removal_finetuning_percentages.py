"""Genera tablas por porcentaje para experimentos de class removal con porcentajes."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.class_removal_analysis import (
    aggregate_learning_curves,
    collect_percentage_summaries,
    plot_accuracy_and_variability_by_percentage,
    plot_accuracy_and_variability_by_percentage_per_dataset,
    plot_mean_learning_curves_by_percentage,
    save_aggregated_learning_curves,
    save_percentage_summaries,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build percentage-level summary tables and mean learning-curve plots "
            "for class-removal result directories that include train percentages."
        ),
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dirs",
        action="append",
        default=None,
        help=(
            "Directory containing class-removal results with percentage runs. "
            "Repeat the flag to process multiple directories."
        ),
    )
    return parser.parse_args()


def summarize_results_dir(results_dir: Path):
    rows_by_percentage = collect_percentage_summaries(results_dir)
    if not rows_by_percentage:
        raise FileNotFoundError(f"No final_metrics.json files found under '{results_dir}'")

    save_percentage_summaries(results_dir, rows_by_percentage)
    aggregated_curves = aggregate_learning_curves(rows_by_percentage)
    save_aggregated_learning_curves(results_dir, aggregated_curves)
    learning_curve_plot = plot_mean_learning_curves_by_percentage(results_dir, aggregated_curves)
    percentage_plots = plot_accuracy_and_variability_by_percentage(results_dir, rows_by_percentage)
    dataset_plots = plot_accuracy_and_variability_by_percentage_per_dataset(results_dir, rows_by_percentage)
    return learning_curve_plot, percentage_plots, dataset_plots


def main():
    args = parse_args()
    results_dirs = args.results_dirs or [
        str(RESULTS_DIR / "class_removal_finetuning"),
        str(RESULTS_DIR / "class_removal_frozen_backbone_head"),
    ]

    for raw_dir in results_dirs:
        results_dir = Path(raw_dir)
        learning_curve_plot, percentage_plots, dataset_plots = summarize_results_dir(results_dir)

        print("Summary tables generated under:")
        print(f"  {results_dir / 'percentage_summaries'}")
        if learning_curve_plot is not None:
            print("Mean learning-curve plot saved at:")
            print(f"  {learning_curve_plot}")
        if percentage_plots:
            print("Percentage summary plots saved at:")
            for plot_path in percentage_plots:
                print(f"  {plot_path}")
        if dataset_plots:
            print("Dataset-level plots saved at:")
            for plot_path in dataset_plots:
                print(f"  {plot_path}")


if __name__ == "__main__":
    main()
