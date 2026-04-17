from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DEVICE, NUM_WORKERS, PLOTS_DIR
from src.dataset.loaders import DATASET_LOADERS
from src.models import build_mobilenet
from src.training import evaluate, finetune
from src.visualization import visualize_top_confused_pairs


def run_similarity_visualization_for_dataset(ds_name: str, loader_fn, top_k=5, num_samples=5):
    """
    Train one model on full data for a dataset and visualize its top confused pairs.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {ds_name}")
    print(f"{'=' * 60}")

    tr, va, te, classes = loader_fn()
    num_classes = len(classes)

    tr_loader = DataLoader(tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_loader = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training MobileNetV3-Small on full {ds_name} dataset...")
    model = build_mobilenet(num_classes)
    finetune(model, tr_loader, va_loader)

    overall, per_class, cm = evaluate(model, te_loader, num_classes)
    print(f"Test accuracy: {overall:.4f}")
    print(f"Mean per-class accuracy: {per_class.mean():.4f}")

    visualize_top_confused_pairs(
        model=model,
        test_loader=te_loader,
        test_dataset=te,
        classes=classes,
        cm=cm,
        device=DEVICE,
        top_k=top_k,
        num_samples=num_samples,
        save_dir=PLOTS_DIR,
        ds_name=ds_name,
    )


def run_all_similarity_visualizations(top_k=5, num_samples=5):
    """
    Train and visualize confusion pairs for every configured dataset.
    """
    print(f"Device: {DEVICE}")

    for ds_name, loader_fn in DATASET_LOADERS.items():
        try:
            run_similarity_visualization_for_dataset(
                ds_name,
                loader_fn,
                top_k=top_k,
                num_samples=num_samples,
            )
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")

    print("\nSimilarity visualization run complete.")


if __name__ == "__main__":
    run_all_similarity_visualizations()
