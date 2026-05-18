"""Compara variantes few-shot y enfrenta la mejor contra otros metodos de adaptacion."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments_config.config import RESULTS_DIR
from src.analysis.fewshot_vs_adaptation_comparison import (
    filter_completed_rows,
    load_all_rows,
    save_comparison_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare prototypical few-shot variants (shots=1/5/10) and contrast "
            "the best-accuracy few-shot setup against adaptation methods after class removal."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Root results directory containing class_removal_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "fewshot_vs_adaptation_comparison"),
        help="Directory where comparison tables, markdown summary and plots will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    all_rows = load_all_rows(results_root)
    if not all_rows:
        raise FileNotFoundError(
            f"No final_metrics.json files found under '{results_root}' for the configured methods."
        )

    completed_rows = filter_completed_rows(all_rows)
    artifacts = save_comparison_outputs(output_dir, all_rows, completed_rows)

    print("Comparison outputs saved under:")
    print(f"  {output_dir}")
    print("Best few-shot variant:")
    print(f"  {artifacts['best_fewshot_variant']}")
    print("Markdown report:")
    print(f"  {artifacts['report_path']}")
    if artifacts["plot_fewshot_path"] is not None:
        print("Few-shot summary plot:")
        print(f"  {artifacts['plot_fewshot_path']}")
    if artifacts["plot_comparison_path"] is not None:
        print("Best few-shot vs adaptation plot:")
        print(f"  {artifacts['plot_comparison_path']}")
    if artifacts["plot_landscape_path"] is not None:
        print("Accuracy vs time landscape plot:")
        print(f"  {artifacts['plot_landscape_path']}")
    if artifacts["plot_deltas_path"] is not None:
        print("Best few-shot delta plot:")
        print(f"  {artifacts['plot_deltas_path']}")


if __name__ == "__main__":
    main()
