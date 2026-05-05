"""Generate evidence plots from the logits-vs-embeddings comparison CSV files."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments_config.config import PLOTS_DIR, RESULTS_DIR


LOGITS_SUMMARY_CSV = RESULTS_DIR / "logits_vs_embeddings_distance_summary.csv"
LOGITS_COMPARISON_CSV = RESULTS_DIR / "logits_vs_embeddings_distance_comparison.csv"
EMBEDDING_SUMMARY_CSV = RESULTS_DIR / "embedding_distance_comparison_summary.csv"
OUTPUT_DIR = PLOTS_DIR / "logits_vs_embeddings_evidence"

DISTANCE_ORDER = [
    "cosine",
    "euclidean",
    "euclidean_l2_normalized",
    "squared_euclidean_l2_normalized",
    "euclidean_normalized",
]


def load_csv_rows(path: Path) -> list:
    """Load a CSV file into a list of dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict, key: str) -> float:
    """Convert a CSV value to float."""
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def ordered_distances(values) -> list:
    """Sort distances with a stable preferred order."""
    seen = list(dict.fromkeys(values))
    fallback = [value for value in seen if value not in DISTANCE_ORDER]
    return [value for value in DISTANCE_ORDER if value in seen] + sorted(fallback)


def ensure_output_dir():
    """Create the output directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str):
    """Persist a matplotlib figure."""
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {path}")


def aggregate_representation_wins(comparison_rows: list):
    """Count how often logits or embeddings win for each distance."""
    distances = ordered_distances(row["distance"] for row in comparison_rows)
    wins = {
        "logits": [],
        "embeddings": [],
    }

    for distance in distances:
        subset = [row for row in comparison_rows if row["distance"] == distance]
        logits_wins = sum(row["better_representation_by_mean_rank"] == "logits" for row in subset)
        embeddings_wins = sum(row["better_representation_by_mean_rank"] == "embeddings" for row in subset)
        wins["logits"].append(logits_wins)
        wins["embeddings"].append(embeddings_wins)

    return distances, wins


def plot_representation_win_counts(comparison_rows: list):
    """Plot how often logits beat embeddings for each distance."""
    distances, wins = aggregate_representation_wins(comparison_rows)
    x = np.arange(len(distances))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_logits = ax.bar(x - width / 2, wins["logits"], width, label="Logits win", color="#1f77b4")
    bars_embeddings = ax.bar(x + width / 2, wins["embeddings"], width, label="Embeddings win", color="#ff7f0e")

    for bars in (bars_logits, bars_embeddings):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(distances, rotation=20, ha="right")
    ax.set_ylabel("Number of dataset-model wins")
    ax.set_title("Representation wins by distance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "representation_wins_by_distance.png")


def plot_representation_mean_rank(summary_rows: list):
    """Plot the average mean-rank by representation and distance."""
    distances = ordered_distances(row["distance"] for row in summary_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(distances))
    width = 0.36

    for offset, representation, color in [
        (-width / 2, "logits", "#1f77b4"),
        (width / 2, "embeddings", "#ff7f0e"),
    ]:
        means = []
        for distance in distances:
            subset = [
                to_float(row, "mean_rank")
                for row in summary_rows
                if row["representation"] == representation and row["distance"] == distance
            ]
            means.append(float(np.nanmean(subset)))

        bars = ax.bar(x + offset, means, width, label=representation, color=color)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(distances, rotation=20, ha="right")
    ax.set_ylabel("Average mean rank (lower is better)")
    ax.set_title("Average ranking by representation and distance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "representation_mean_rank_by_distance.png")


def plot_representation_metric_boxplots(summary_rows: list):
    """Plot boxplots for numeric metrics comparing logits vs embeddings."""
    metric_columns = [
        ("knn_accuracy_mean", "kNN acc"),
        ("silhouette_score", "Silhouette"),
        ("ratio_intra_inter", "Ratio intra/inter"),
        ("distance_confusion_correlation", "Dist-conf corr"),
        ("mean_centroid_margin", "Margin"),
        ("negative_margin_fraction", "Neg margin frac"),
    ]
    distances = ordered_distances(row["distance"] for row in summary_rows)

    for distance in distances:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, (column, label) in zip(axes, metric_columns):
            logits_values = [
                to_float(row, column)
                for row in summary_rows
                if row["representation"] == "logits" and row["distance"] == distance
            ]
            embeddings_values = [
                to_float(row, column)
                for row in summary_rows
                if row["representation"] == "embeddings" and row["distance"] == distance
            ]

            box = ax.boxplot(
                [logits_values, embeddings_values],
                patch_artist=True,
                labels=["logits", "embeddings"],
            )
            colors = ["#1f77b4", "#ff7f0e"]
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.78)

            for median in box["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(
            f"Numeric metric distributions: logits vs embeddings\nDistance = {distance}",
            fontsize=12,
            fontweight="bold",
        )
        save_figure(fig, f"boxplot_numeric_metrics_logits_vs_embeddings_{distance}.png")


def plot_metric_advantage_boxplots(comparison_rows: list):
    """Plot boxplots of logits advantage for each metric across distances."""
    metric_columns = [
        ("knn_accuracy_advantage_embeddings", "kNN acc"),
        ("silhouette_advantage_embeddings", "Silhouette"),
        ("ratio_intra_inter_advantage_embeddings", "Ratio intra/inter"),
        ("distance_confusion_corr_abs_advantage_embeddings", "Dist-conf corr"),
        ("mean_centroid_margin_advantage_embeddings", "Margin"),
        ("negative_margin_fraction_advantage_embeddings", "Neg margin frac"),
    ]
    distances = ordered_distances(row["distance"] for row in comparison_rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, (column, label) in zip(axes, metric_columns):
        data = []
        for distance in distances:
            values = [
                -to_float(row, column)
                for row in comparison_rows
                if row["distance"] == distance
            ]
            data.append(values)

        box = ax.boxplot(data, patch_artist=True, labels=distances)
        palette = plt.cm.Blues(np.linspace(0.45, 0.85, len(distances)))
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Distribution of logits advantage by metric and distance\n(positive means logits are better)",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, "boxplot_logits_advantage_by_metric.png")


def plot_logits_advantage_heatmap(comparison_rows: list):
    """Plot a heatmap of average logits advantage across metrics and distances."""
    metric_columns = [
        ("knn_accuracy_advantage_embeddings", "kNN acc"),
        ("silhouette_advantage_embeddings", "Silhouette"),
        ("ratio_intra_inter_advantage_embeddings", "Ratio intra/inter"),
        ("distance_confusion_corr_abs_advantage_embeddings", "Dist-conf corr"),
        ("mean_centroid_margin_advantage_embeddings", "Margin"),
        ("negative_margin_fraction_advantage_embeddings", "Neg margin frac"),
    ]
    distances = ordered_distances(row["distance"] for row in comparison_rows)
    matrix = np.zeros((len(metric_columns), len(distances)), dtype=float)

    for row_idx, (column, _) in enumerate(metric_columns):
        for col_idx, distance in enumerate(distances):
            values = [
                -to_float(row, column)
                for row in comparison_rows
                if row["distance"] == distance
            ]
            matrix[row_idx, col_idx] = float(np.nanmean(values))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    vmax = np.nanmax(np.abs(matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    image = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(image, ax=ax, label="Average logits advantage")
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels(distances, rotation=20, ha="right")
    ax.set_yticks(range(len(metric_columns)))
    ax.set_yticklabels([label for _, label in metric_columns])
    ax.set_title("Where logits gain or lose against embeddings")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[i, j]) > vmax * 0.45 else "black",
            )

    save_figure(fig, "logits_advantage_heatmap.png")


def best_distance_counts(rows: list, representation_key: str = None) -> dict:
    """Count how often each distance is the best one by mean rank."""
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["architecture"])
        if representation_key is not None:
            key = key + (row["representation"],)
            if row["representation"] != representation_key:
                continue
        grouped.setdefault(key, []).append(row)

    counts = {}
    for subset in grouped.values():
        best_row = min(subset, key=lambda item: to_float(item, "mean_rank"))
        counts[best_row["distance"]] = counts.get(best_row["distance"], 0) + 1

    return counts


def plot_best_distance_counts(summary_rows: list, embedding_rows: list):
    """Plot how often each distance wins overall and inside each representation study."""
    logits_counts = best_distance_counts(summary_rows, representation_key="logits")
    embeddings_counts = best_distance_counts(summary_rows, representation_key="embeddings")
    embedding_only_counts = best_distance_counts(embedding_rows, representation_key=None)

    distances = ordered_distances(
        list(logits_counts.keys()) + list(embeddings_counts.keys()) + list(embedding_only_counts.keys())
    )
    x = np.arange(len(distances))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ("Logits", logits_counts, "#1f77b4", -width),
        ("Embeddings", embeddings_counts, "#ff7f0e", 0.0),
        ("Embeddings-only study", embedding_only_counts, "#2ca02c", width),
    ]

    for label, counts, color, offset in series:
        values = [counts.get(distance, 0) for distance in distances]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(distances, rotation=20, ha="right")
    ax.set_ylabel("Times selected as best distance")
    ax.set_title("Best distance winner counts")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "best_distance_winner_counts.png")


def plot_cosine_vs_others(summary_rows: list, embedding_rows: list):
    """Plot how often cosine beats the other distances by average mean rank."""
    def counts_against_others(rows: list, representation: str = None):
        grouped = {}
        for row in rows:
            if representation is not None and row.get("representation") != representation:
                continue
            key = (row["dataset"], row["architecture"])
            if representation is not None:
                key = key + (representation,)
            grouped.setdefault(key, []).append(row)

        total = 0
        cosine_best = 0
        for subset in grouped.values():
            distances_present = {row["distance"] for row in subset}
            if "cosine" not in distances_present or len(distances_present) < 2:
                continue
            total += 1
            best_row = min(subset, key=lambda item: to_float(item, "mean_rank"))
            if best_row["distance"] == "cosine":
                cosine_best += 1
        return cosine_best, total

    values = [
        counts_against_others(summary_rows, "logits"),
        counts_against_others(summary_rows, "embeddings"),
        counts_against_others(embedding_rows, None),
    ]
    labels = ["Logits", "Embeddings", "Embeddings-only study"]
    fractions = [wins / total if total else 0.0 for wins, total in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fractions, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for bar, (wins, total), fraction in zip(bars, values, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(fraction + 0.03, 1.02),
            f"{wins}/{total}\n{fraction:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of cases where cosine is best")
    ax.set_title("How often cosine wins")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "cosine_win_fraction.png")


def write_text_summary(summary_rows: list, comparison_rows: list, embedding_rows: list):
    """Write a compact text summary next to the plots."""
    total_comparisons = len(comparison_rows)
    logits_wins = sum(row["better_representation_by_mean_rank"] == "logits" for row in comparison_rows)
    embeddings_wins = sum(row["better_representation_by_mean_rank"] == "embeddings" for row in comparison_rows)

    logits_best_cosine, logits_total = _cosine_stats(summary_rows, "logits")
    emb_best_cosine, emb_total = _cosine_stats(summary_rows, "embeddings")
    emb_only_best_cosine, emb_only_total = _cosine_stats(embedding_rows, None)

    path = OUTPUT_DIR / "summary.txt"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Evidence summary\n")
        handle.write("================\n\n")
        handle.write(f"Total logits-vs-embeddings comparisons: {total_comparisons}\n")
        handle.write(f"Logits wins by mean rank: {logits_wins}\n")
        handle.write(f"Embeddings wins by mean rank: {embeddings_wins}\n\n")
        handle.write(f"Cosine best in logits rows: {logits_best_cosine}/{logits_total}\n")
        handle.write(f"Cosine best in embeddings rows: {emb_best_cosine}/{emb_total}\n")
        handle.write(f"Cosine best in embeddings-only study: {emb_only_best_cosine}/{emb_only_total}\n")
    print(f"Summary saved -> {path}")


def _cosine_stats(rows: list, representation: str = None):
    """Compute cosine win counts."""
    grouped = {}
    for row in rows:
        if representation is not None and row.get("representation") != representation:
            continue
        key = (row["dataset"], row["architecture"])
        if representation is not None:
            key = key + (representation,)
        grouped.setdefault(key, []).append(row)

    total = 0
    wins = 0
    for subset in grouped.values():
        total += 1
        best_row = min(subset, key=lambda item: to_float(item, "mean_rank"))
        if best_row["distance"] == "cosine":
            wins += 1
    return wins, total


def main():
    """Load result CSVs and generate combined evidence plots."""
    ensure_output_dir()
    logits_summary_rows = load_csv_rows(LOGITS_SUMMARY_CSV)
    logits_comparison_rows = load_csv_rows(LOGITS_COMPARISON_CSV)
    embedding_summary_rows = load_csv_rows(EMBEDDING_SUMMARY_CSV)

    plot_representation_win_counts(logits_comparison_rows)
    plot_representation_mean_rank(logits_summary_rows)
    plot_representation_metric_boxplots(logits_summary_rows)
    plot_metric_advantage_boxplots(logits_comparison_rows)
    plot_logits_advantage_heatmap(logits_comparison_rows)
    plot_best_distance_counts(logits_summary_rows, embedding_summary_rows)
    plot_cosine_vs_others(logits_summary_rows, embedding_summary_rows)
    write_text_summary(logits_summary_rows, logits_comparison_rows, embedding_summary_rows)

    print("\nEvidence plots completed.")


if __name__ == "__main__":
    main()
