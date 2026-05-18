"""
reduce_and_evaluate.py
======================
Studies how classification accuracy degrades when training examples are reduced.

Two reduction strategies are implemented:

1. reduce_all_classes(dataset, fraction)
   Randomly removes (1 - fraction) × 100 % of examples from EVERY class.

2. reduce_least_confused_class(dataset, classes, confusion_matrix, fraction)
   Removes examples only from the class that has the LOWEST off-diagonal
   confusion rate — i.e. the class the model has "most confidently learned"
   and that therefore may need fewer examples.

Both functions return a Subset of the original dataset.

The main experiment function  run_reduction_study()  iterates over a list of
retention fractions, fine-tunes only the classification head (frozen backbone)
from scratch each time, and records per-class and overall accuracy.

Model used: MobileNetV3-Small  (lightest, best accuracy in prior experiments)

Usage
-----
    python reduce_and_evaluate.py

Outputs
-------
    results/reduction_all_classes.csv
    results/reduction_least_confused.csv
    (plots are saved to results/plots/)
"""

import time
from torch.utils.data import DataLoader

from src.experiments_config.config import BATCH_SIZE, DEVICE, NUM_WORKERS, PLOTS_DIR, RESULTS_DIR, RETENTION_FRACTIONS
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class
from src.models import build_mobilenet
from src.core.reduction import find_least_confused_class, reduce_all_classes, reduce_least_confused_class
from src.core.training import evaluate, finetune
from src.core.visualization import (
    plot_class_examples_bar,
    plot_least_confused_reduction,
    plot_overall_accuracy_vs_fraction,
    plot_per_class_accuracy_heatmap,
    visualize_top_confused_pairs,
    write_csv,
)

print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runners
# ─────────────────────────────────────────────────────────────────────────────
def run_reduction_all_classes(ds_name: str, tr_full, va_split, te_split,
                               classes: list, fractions=RETENTION_FRACTIONS):
    """
    Strategy 1: uniformly reduce all classes.

    For each retention fraction:
      - subsample training data
      - build a fresh MobileNetV3-Small head
      - fine-tune for EPOCHS epochs
      - evaluate on the fixed test set

    Returns list of result dicts (one per fraction).
    """
    num_classes = len(classes)
    rows = []

    # Build fixed val/test loaders (never reduced)
    va_loader = DataLoader(va_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n{'─'*60}")
    print(f"  [{ds_name}] Strategy: Reduce ALL classes")
    print(f"{'─'*60}")

    # Count examples at full size
    full_counts = count_examples_per_class(tr_full, classes)
    plot_class_examples_bar(
        full_counts, ds_name,
        save_path=PLOTS_DIR / f"{ds_name.replace(' ', '_')}_examples_per_class.png",
    )

    for frac in fractions:
        print(f"\n  Fraction kept: {frac:.0%}")

        subset    = reduce_all_classes(tr_full, fraction=frac)
        tr_loader = DataLoader(subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        counts = count_examples_per_class(subset, classes)
        total  = sum(counts.values())
        print(f"  Total training examples: {total}  "
              f"(per class min={min(counts.values())} max={max(counts.values())})")

        model = build_mobilenet(num_classes)
        t0    = time.time()
        finetune(model, tr_loader, va_loader)
        elapsed = time.time() - t0

        overall, per_cls, _ = evaluate(model, te_loader, num_classes)
        print(f"  Test acc: {overall:.4f}  time: {elapsed:.1f}s")

        row = {
            "dataset":          ds_name,
            "fraction":         frac,
            "total_train":      total,
            "overall_test_acc": round(overall, 6),
            "elapsed_seconds":  round(elapsed, 2),
        }
        for i, name in enumerate(classes):
            row[f"per_class_acc_{i}"] = round(float(per_cls[i]), 6)
            row[f"n_train_{name}"]    = counts[name]
        rows.append(row)
        del model

    return rows


def run_reduction_least_confused(ds_name: str, tr_full, va_split, te_split,
                                  classes: list, fractions=RETENTION_FRACTIONS):
    """
    Strategy 2: reduce only the least-confused class.

    First runs a full-data evaluation to identify the least-confused class,
    then iterates over retention fractions applied only to that class.

    Returns list of result dicts (one per fraction) + the least-confused class name.
    """
    num_classes = len(classes)
    rows = []

    va_loader = DataLoader(va_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    tr_loader_full = DataLoader(tr_full, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print(f"\n{'─'*60}")
    print(f"  [{ds_name}] Strategy: Reduce LEAST-CONFUSED class")
    print(f"{'─'*60}")

    # ── Step 1: train once on full data to get a confusion matrix ──────────
    print("  Training on full data to identify least-confused class...")
    model_full = build_mobilenet(num_classes)
    finetune(model_full, tr_loader_full, va_loader)
    _, _, cm_full = evaluate(model_full, te_loader, num_classes)
    del model_full

    lc_idx  = find_least_confused_class(cm_full)
    lc_name = classes[lc_idx]
    print(f"  Least-confused class: '{lc_name}' (index {lc_idx})")

    # ── Step 2: iterate over fractions ────────────────────────────────────
    for frac in fractions:
        print(f"\n  Fraction kept for '{lc_name}': {frac:.0%}")

        subset    = reduce_least_confused_class(tr_full, cm_full, fraction=frac)
        tr_loader = DataLoader(subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        counts = count_examples_per_class(subset, classes)
        total  = sum(counts.values())
        print(f"  '{lc_name}' examples: {counts[lc_name]}  |  "
              f"total training: {total}")

        model = build_mobilenet(num_classes)
        t0    = time.time()
        finetune(model, tr_loader, va_loader)
        elapsed = time.time() - t0

        overall, per_cls, cm = evaluate(model, te_loader, num_classes)
        
        if frac == 1.0:
            visualize_top_confused_pairs(
                model=model,
                test_loader=te_loader,
                test_dataset=te_split,
                classes=classes,
                cm=cm,
                device=DEVICE,
                top_k=5,
                num_samples=5,
                save_dir=PLOTS_DIR,
                ds_name=ds_name,
            )
        
        
        print(f"  Test acc: {overall:.4f}  time: {elapsed:.1f}s")

        row = {
            "dataset":              ds_name,
            "least_confused_class": lc_name,
            "fraction":             frac,
            "n_lc_train":           counts[lc_name],
            "total_train":          total,
            "overall_test_acc":     round(overall, 6),
            "elapsed_seconds":      round(elapsed, 2),
        }
        for i, name in enumerate(classes):
            row[f"per_class_acc_{i}"] = round(float(per_cls[i]), 6)
        rows.append(row)
        del model

    return rows, lc_name, cm_full
    
    



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    all_rows_all     = []   # all-classes reduction
    all_rows_lc      = []   # least-confused reduction
    results_by_ds_all = {}  # for the cross-dataset plot

    for ds_name, loader_fn in DATASET_LOADERS.items():
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        try:
            tr, va, te, classes = loader_fn()
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # ── Strategy 1: all classes ──────────────────────────────────────
        rows_all = run_reduction_all_classes(ds_name, tr, va, te, classes)
        results_by_ds_all[ds_name] = rows_all
        all_rows_all.extend(rows_all)

        plot_per_class_accuracy_heatmap(
            rows_all, classes, ds_name,
            strategy_label="All-class uniform reduction",
            save_path=PLOTS_DIR / f"{ds_name.replace(' ', '_')}_all_classes_heatmap.png",
        )

        # ── Strategy 2: least-confused class ────────────────────────────
        rows_lc, lc_name, _ = run_reduction_least_confused(ds_name, tr, va, te, classes)
        all_rows_lc.extend(rows_lc)

        plot_least_confused_reduction(
            rows_lc, classes, lc_name, ds_name,
            save_path=PLOTS_DIR / f"{ds_name.replace(' ', '_')}_least_confused.png",
        )

    # ── Cross-dataset overview plot ───────────────────────────────────────
    if results_by_ds_all:
        plot_overall_accuracy_vs_fraction(
            results_by_ds_all,
            strategy_label="All-class uniform reduction – MobileNetV3-Small",
            save_path=PLOTS_DIR / "all_datasets_overall_accuracy.png",
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────
    if all_rows_all:
        fields_all = list(all_rows_all[0].keys())
        write_csv(
            RESULTS_DIR / "reduction_all_classes.csv",
            all_rows_all, fields_all
        )

    if all_rows_lc:
        fields_lc = list(all_rows_lc[0].keys())
        write_csv(
            RESULTS_DIR / "reduction_least_confused.csv",
            all_rows_lc, fields_lc
        )

    print("\nAll done.")


if __name__ == "__main__":
    run_all()
