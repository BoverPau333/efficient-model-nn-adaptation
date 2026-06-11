"""Genera graficas y tablas por porcentaje para class addition con fine-tuning."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.class_addition_percentage_analysis import (
    aggregate_learning_curves,
    build_accuracy_drop_variability_rows,
    collect_percentage_summaries,
    plot_accuracy_and_variability_by_percentage,
    plot_accuracy_and_variability_by_percentage_per_dataset,
    plot_mean_learning_curves_by_percentage,
    save_accuracy_drop_variability_artifacts,
    save_aggregated_learning_curves,
    save_percentage_summaries,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build percentage-level tables and plots for class-addition fine-tuning results."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR / "class_addition_finetuning_head_only"),
        help="Directory containing class-addition percentage results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    rows_by_percentage = collect_percentage_summaries(results_dir)
    if not rows_by_percentage:
        raise FileNotFoundError(f"No final_metrics.json files found under '{results_dir}'")

    save_percentage_summaries(results_dir, rows_by_percentage)

    aggregated_curves = aggregate_learning_curves(rows_by_percentage)
    save_aggregated_learning_curves(results_dir, aggregated_curves)

    learning_curve_plot = plot_mean_learning_curves_by_percentage(results_dir, aggregated_curves)
    percentage_plots = plot_accuracy_and_variability_by_percentage(results_dir, rows_by_percentage)
    dataset_plots = plot_accuracy_and_variability_by_percentage_per_dataset(results_dir, rows_by_percentage)

    summary_rows = build_accuracy_drop_variability_rows(rows_by_percentage)
    table_paths = save_accuracy_drop_variability_artifacts(results_dir, summary_rows)

    print("Summary tables generated under:")
    print(f"  {results_dir / 'percentage_summaries'}")
    print("Analysis artifacts generated under:")
    print(f"  {results_dir / 'analysis'}")
    if learning_curve_plot is not None:
        print("Learning-curve plot:")
        print(f"  {learning_curve_plot}")
    if percentage_plots:
        print("Overall percentage plots:")
        for plot_path in percentage_plots:
            print(f"  {plot_path}")
    if dataset_plots:
        print("Dataset-level plots:")
        for plot_path in dataset_plots:
            print(f"  {plot_path}")
    if table_paths:
        print("Accuracy/variability table:")
        for artifact_path in table_paths.values():
            print(f"  {artifact_path}")


if __name__ == "__main__":
    main()
