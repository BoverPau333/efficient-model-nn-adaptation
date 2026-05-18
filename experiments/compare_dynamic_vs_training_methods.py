"""Compara dynamic fine-tuning porc_10 con otros metodos de entrenamiento."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.dynamic_training_method_comparison import (
    load_method_rows,
    plot_dataset_comparison,
    plot_overall_comparison,
    save_comparison_tables,
)
from src.experiments_config.config import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare dynamic fine-tuning against other training methods for a fixed train percentage.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Root results directory.",
    )
    parser.add_argument(
        "--train-percentage",
        type=float,
        default=10.0,
        help="Train percentage to compare across methods.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "dynamic_training_method_comparison" / "porc_10"),
        help="Directory where comparison tables and plots will be written.",
    )
    parser.add_argument(
        "--dynamic-variant",
        default=None,
        help="Subcarpeta opcional dentro de los metodos dynamic, por ejemplo 'early_stopping', o 'both' para comparar normal vs early stopping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    rows = load_method_rows(
        results_root,
        train_percentage=args.train_percentage,
        dynamic_variant=args.dynamic_variant,
    )
    if not rows:
        raise FileNotFoundError(
            f"No results found under '{results_root}' for train_percentage={args.train_percentage}."
        )

    summaries = save_comparison_tables(output_dir, rows)
    overall_plot = plot_overall_comparison(output_dir, summaries["overall"])
    dataset_plot = plot_dataset_comparison(output_dir, summaries["by_dataset"])

    print("Comparison tables saved under:")
    print(f"  {output_dir}")
    print("Plots:")
    print(f"  {overall_plot}")
    if dataset_plot is not None:
        print(f"  {dataset_plot}")


if __name__ == "__main__":
    main()
