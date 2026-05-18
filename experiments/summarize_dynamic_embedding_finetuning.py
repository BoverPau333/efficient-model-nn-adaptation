"""Genera tablas y graficas de analisis para dynamic embedding fine-tuning."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.dynamic_embedding_analysis import (
    aggregate_learning_curves,
    build_method_dataset_summary,
    collect_dynamic_results,
    plot_examples_used_by_dataset,
    plot_learning_curves_by_dataset,
    plot_mean_learning_curves_overall,
    plot_method_status_counts,
    plot_overfitting_gap_overall,
    save_aggregated_learning_curves,
    save_dynamic_result_tables,
    save_method_dataset_summary,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build analysis artifacts for dynamic embedding fine-tuning results.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR / "dynamic_embedding_finetuning"),
        help="Directorio raiz con los resultados dinamicos.",
    )
    parser.add_argument(
        "--train-percentage",
        type=float,
        default=None,
        help="Filtra el analisis para incluir solo ejecuciones con este porcentaje de train.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Filtra solo resultados cuya ruta contenga esta subcarpeta, por ejemplo 'early_stopping'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_dir)
    analysis_dir = results_root / "analysis"
    if args.variant:
        analysis_dir = analysis_dir / args.variant
    if args.train_percentage is not None:
        percentage_label = str(int(args.train_percentage)) if float(args.train_percentage).is_integer() else str(args.train_percentage)
        analysis_dir = analysis_dir / f"porc_{percentage_label}"

    completed_rows, failed_rows = collect_dynamic_results(
        results_root,
        train_percentage=args.train_percentage,
        variant=args.variant,
    )
    if not completed_rows and not failed_rows:
        raise FileNotFoundError(f"No dynamic embedding results found under '{results_root}'")

    save_dynamic_result_tables(analysis_dir, completed_rows, failed_rows)
    summary_rows = build_method_dataset_summary(completed_rows, failed_rows)
    summary_path = save_method_dataset_summary(analysis_dir, summary_rows)

    aggregated_overall = aggregate_learning_curves(completed_rows, group_by=None)
    aggregated_by_dataset = aggregate_learning_curves(completed_rows, group_by="dataset")
    save_aggregated_learning_curves(analysis_dir, aggregated_overall, "mean_learning_curve_overall")
    save_aggregated_learning_curves(analysis_dir, aggregated_by_dataset, "mean_learning_curve_dataset")

    plot_paths = []
    plot_paths.extend(plot_mean_learning_curves_overall(analysis_dir, aggregated_overall))
    overfit_path = plot_overfitting_gap_overall(analysis_dir, aggregated_overall)
    if overfit_path is not None:
        plot_paths.append(overfit_path)
    plot_paths.extend(plot_learning_curves_by_dataset(analysis_dir, aggregated_by_dataset))
    examples_path = plot_examples_used_by_dataset(analysis_dir, completed_rows)
    if examples_path is not None:
        plot_paths.append(examples_path)
    status_path = plot_method_status_counts(analysis_dir, completed_rows, failed_rows)
    if status_path is not None:
        plot_paths.append(status_path)

    print("Dynamic embedding analysis generated under:")
    print(f"  {analysis_dir}")
    if summary_path is not None:
        print("Method/dataset summary:")
        print(f"  {summary_path}")
    if plot_paths:
        print("Plots:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
