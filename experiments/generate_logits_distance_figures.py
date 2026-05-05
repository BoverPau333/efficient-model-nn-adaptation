"""Generate TFG-ready comparative figures for logits vs embeddings distance studies."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments_config.config import RESULTS_DIR


DEFAULT_SUMMARY_CSV = Path("/mnt/data/logits_vs_embeddings_distance_summary.csv")
DEFAULT_COMPARISON_CSV = Path("/mnt/data/logits_vs_embeddings_distance_comparison.csv")
DEFAULT_OUTPUT_DIR = Path("/mnt/data/figures_logits_distance/")

FALLBACK_SUMMARY_CSV = RESULTS_DIR / "logits_vs_embeddings_distance_summary.csv"
FALLBACK_COMPARISON_CSV = RESULTS_DIR / "logits_vs_embeddings_distance_comparison.csv"
FALLBACK_OUTPUT_DIR = RESULTS_DIR / "figures_logits_distance"

DISTANCE_ORDER = [
    "cosine",
    "euclidean",
    "euclidean_l2_normalized",
    "squared_euclidean_l2_normalized",
]
REPRESENTATION_ORDER = ["logits", "embeddings"]
METRIC_SPECS = [
    ("knn_accuracy_mean", "kNN accuracy", "higher"),
    ("silhouette_score", "Silhouette score", "higher"),
    ("ratio_intra_inter", "Intra/inter ratio", "lower"),
    ("distance_confusion_correlation", "Distance-confusion corr.", "higher"),
    ("mean_centroid_margin", "Mean centroid margin", "higher"),
    ("negative_margin_fraction", "Negative margin fraction", "lower"),
]
ADVANTAGE_COLUMNS = [
    ("knn_accuracy_advantage_embeddings", "kNN accuracy"),
    ("silhouette_advantage_embeddings", "Silhouette score"),
    ("ratio_intra_inter_advantage_embeddings", "Intra/inter ratio"),
    ("distance_confusion_corr_abs_advantage_embeddings", "Distance-confusion corr."),
    ("mean_centroid_margin_advantage_embeddings", "Mean centroid margin"),
    ("negative_margin_fraction_advantage_embeddings", "Negative margin fraction"),
]


def _pick_existing_path(primary: Path, fallback: Path) -> Path:
    """Return the preferred path when available, otherwise a local fallback."""
    if primary.is_file():
        return primary
    if fallback.is_file():
        print(f"[INFO] Using fallback input instead of missing {primary}: {fallback}")
        return fallback
    raise FileNotFoundError(f"Neither {primary} nor {fallback} exists.")


def _pick_output_dir(primary: Path, fallback: Path) -> Path:
    """Return the preferred output directory when writable, otherwise a local fallback."""
    parent = primary.parent
    if parent.exists() and parent.is_dir():
        return primary
    print(f"[INFO] Using fallback output directory instead of unavailable {primary}: {fallback}")
    return fallback


def load_csv_rows(path: Path) -> list:
    """Load a CSV file into dictionaries."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict, key: str) -> float:
    """Parse a float value from a CSV row."""
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def mean_or_nan(values: list) -> float:
    """Compute a nan-safe mean."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def ordered_distances(rows: list) -> list:
    """Return distances in a stable order."""
    present = {row["distance"] for row in rows}
    return [distance for distance in DISTANCE_ORDER if distance in present]


def combination_label(representation: str, distance: str) -> str:
    """Build a readable representation+distance label."""
    return f"{representation} | {distance}"


def save_figure(fig, output_dir: Path, filename: str):
    """Save a figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved {png_path.name} and {pdf_path.name}")


def build_matrix(rows: list, value_key: str):
    """Aggregate a representation-by-distance matrix for a numeric field."""
    distances = ordered_distances(rows)
    matrix = np.zeros((len(REPRESENTATION_ORDER), len(distances)), dtype=float)
    for i, representation in enumerate(REPRESENTATION_ORDER):
        for j, distance in enumerate(distances):
            values = [
                to_float(row, value_key)
                for row in rows
                if row["representation"] == representation and row["distance"] == distance
            ]
            matrix[i, j] = mean_or_nan(values)
    return distances, matrix


def plot_heatmap_rank(summary_rows: list, value_key: str, title: str, filename: str, output_dir: Path):
    """Plot annotated rank heatmaps."""
    distances, matrix = build_matrix(summary_rows, value_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(matrix, cmap="YlGn_r", aspect="auto")
    plt.colorbar(image, ax=ax, label=f"Average {value_key}")
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels(distances, rotation=20, ha="right")
    ax.set_yticks(range(len(REPRESENTATION_ORDER)))
    ax.set_yticklabels(REPRESENTATION_ORDER)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    save_figure(fig, output_dir, filename)
    best_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    print(
        f"[INTERPRETATION] {title}: best average rank is "
        f"{REPRESENTATION_ORDER[best_idx[0]]} with {distances[best_idx[1]]} "
        f"({matrix[best_idx]:.2f})."
    )


def plot_slope_charts(summary_rows: list, output_dir: Path):
    """Plot logits-vs-embeddings slope charts for each distance."""
    distances = ordered_distances(summary_rows)
    pair_keys = sorted({(row["dataset"], row["architecture"]) for row in summary_rows})

    for distance in distances:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        subset = [row for row in summary_rows if row["distance"] == distance]

        for ax, (metric_key, metric_label, direction) in zip(axes, METRIC_SPECS):
            for dataset_name, architecture in pair_keys:
                pair_rows = [
                    row for row in subset
                    if row["dataset"] == dataset_name and row["architecture"] == architecture
                ]
                if len(pair_rows) != 2:
                    continue
                row_by_representation = {row["representation"]: row for row in pair_rows}
                y_values = [
                    to_float(row_by_representation["logits"], metric_key),
                    to_float(row_by_representation["embeddings"], metric_key),
                ]
                color = "#1f77b4" if y_values[0] >= y_values[1] else "#ff7f0e"
                if direction == "lower":
                    color = "#1f77b4" if y_values[0] <= y_values[1] else "#ff7f0e"
                ax.plot([0, 1], y_values, color=color, alpha=0.45, linewidth=1.6)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["logits", "embeddings"])
            title = metric_label
            if direction == "lower":
                ax.invert_yaxis()
                title += " (lower is better; axis inverted)"
            else:
                title += " (higher is better)"
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(f"Slope charts by metric for distance = {distance}", fontsize=13, fontweight="bold")
        save_figure(fig, output_dir, f"slope_charts_{distance}")
        print(
            f"[INTERPRETATION] Slope charts for {distance}: blue lines favor logits, orange lines favor embeddings."
        )


def plot_logits_win_heatmap(comparison_rows: list, output_dir: Path):
    """Plot the percentage of cases where logits beat embeddings."""
    distances = ordered_distances(comparison_rows)
    matrix = np.zeros((len(distances), len(ADVANTAGE_COLUMNS)), dtype=float)

    for i, distance in enumerate(distances):
        subset = [row for row in comparison_rows if row["distance"] == distance]
        for j, (column, _) in enumerate(ADVANTAGE_COLUMNS):
            wins = [to_float(row, column) < 0 for row in subset]
            matrix[i, j] = 100.0 * (sum(wins) / len(wins)) if wins else float("nan")

    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(image, ax=ax, label="% of cases where logits win")
    ax.set_xticks(range(len(ADVANTAGE_COLUMNS)))
    ax.set_xticklabels([label for _, label in ADVANTAGE_COLUMNS], rotation=20, ha="right")
    ax.set_yticks(range(len(distances)))
    ax.set_yticklabels(distances)
    ax.set_title("Logits win rate by distance and metric")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9, color="black")

    save_figure(fig, output_dir, "heatmap_logits_win_percentage")
    best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    print(
        f"[INTERPRETATION] Logits win-rate heatmap: strongest cell is {distances[best_idx[0]]} on "
        f"{ADVANTAGE_COLUMNS[best_idx[1]][1]} with {matrix[best_idx]:.1f}% logits wins."
    )


def plot_rank_distributions(summary_rows: list, output_dir: Path):
    """Plot aggregated and representation-separated boxplots for rank variables."""
    distances = ordered_distances(summary_rows)

    for rank_key in ["mean_rank", "overall_rank"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [
            [to_float(row, rank_key) for row in summary_rows if row["distance"] == distance]
            for distance in distances
        ]
        box = ax.boxplot(data, patch_artist=True, tick_labels=distances)
        colors = plt.cm.GnBu(np.linspace(0.45, 0.85, len(distances)))
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(f"Aggregated distribution of {rank_key} by distance")
        ax.set_ylabel(rank_key)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, output_dir, f"boxplot_{rank_key}_by_distance")

        fig, axes = plt.subplots(1, len(REPRESENTATION_ORDER), figsize=(12, 4.8), sharey=True)
        for ax, representation in zip(axes, REPRESENTATION_ORDER):
            rep_data = [
                [
                    to_float(row, rank_key)
                    for row in summary_rows
                    if row["distance"] == distance and row["representation"] == representation
                ]
                for distance in distances
            ]
            box = ax.boxplot(rep_data, patch_artist=True, tick_labels=distances)
            colors = plt.cm.Set2(np.linspace(0.2, 0.8, len(distances)))
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_title(f"{representation}")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{rank_key} by distance, separated by representation", fontsize=13, fontweight="bold")
        save_figure(fig, output_dir, f"boxplot_{rank_key}_by_distance_and_representation")

        grouped_means = {
            distance: mean_or_nan([to_float(row, rank_key) for row in summary_rows if row["distance"] == distance])
            for distance in distances
        }
        best_distance = min(grouped_means, key=grouped_means.get)
        print(
            f"[INTERPRETATION] {rank_key} boxplots: {best_distance} has the lowest average {rank_key} "
            f"({grouped_means[best_distance]:.2f})."
        )


def plot_best_configuration_counts(summary_rows: list, output_dir: Path):
    """Count how often each representation+distance is the best configuration."""
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["dataset"], row["architecture"])].append(row)

    counts = Counter()
    for rows in grouped.values():
        best_row = min(rows, key=lambda row: to_float(row, "mean_rank"))
        counts[combination_label(best_row["representation"], best_row["distance"])] += 1

    labels = []
    values = []
    for representation in REPRESENTATION_ORDER:
        for distance in DISTANCE_ORDER:
            label = combination_label(representation, distance)
            if label in counts:
                labels.append(label)
                values.append(counts[label])

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color=plt.cm.tab20(np.linspace(0.05, 0.95, len(labels))))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_title("Best configuration count based on mean rank")
    ax.set_ylabel("Number of dataset-architecture wins")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "bar_best_configuration_count")

    winner = labels[int(np.argmax(values))]
    print(f"[INTERPRETATION] Best-configuration bar chart: most frequent winner is {winner} with {max(values)} wins.")


def rank_values_desc(values_by_key: dict, higher_is_better: bool) -> dict:
    """Assign ranks to combination averages for one metric."""
    items = list(values_by_key.items())
    ordered = sorted(
        items,
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    return {key: rank for rank, (key, _) in enumerate(ordered, start=1)}


def plot_bump_chart(summary_rows: list, output_dir: Path):
    """Plot a bump chart of average ranks by metric for each representation+distance combination."""
    combo_metric_values = defaultdict(list)
    for row in summary_rows:
        combo = combination_label(row["representation"], row["distance"])
        for metric_key, _, _ in METRIC_SPECS:
            combo_metric_values[(combo, metric_key)].append(to_float(row, metric_key))

    combos = [
        combination_label(representation, distance)
        for representation in REPRESENTATION_ORDER
        for distance in DISTANCE_ORDER
    ]

    metric_ranks = {}
    for metric_key, _, direction in METRIC_SPECS:
        averages = {
            combo: mean_or_nan(combo_metric_values[(combo, metric_key)])
            for combo in combos
            if combo_metric_values[(combo, metric_key)]
        }
        metric_ranks[metric_key] = rank_values_desc(averages, higher_is_better=(direction == "higher"))

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(METRIC_SPECS))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(combos)))

    for color, combo in zip(colors, combos):
        y_values = [metric_ranks[metric_key][combo] for metric_key, _, _ in METRIC_SPECS]
        ax.plot(x_positions, y_values, marker="o", linewidth=2, label=combo, color=color, alpha=0.9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label, _ in METRIC_SPECS], rotation=20, ha="right")
    ax.set_yticks(range(1, len(combos) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("Average rank within metric")
    ax.set_title("Bump chart of representation+distance ranks by metric")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save_figure(fig, output_dir, "bump_chart_ranks_by_metric")

    first_places = Counter()
    for metric_key in metric_ranks:
        for combo, rank in metric_ranks[metric_key].items():
            if rank == 1:
                first_places[combo] += 1
    winner = max(first_places, key=first_places.get)
    print(f"[INTERPRETATION] Bump chart: {winner} tops the largest number of metric-specific average rankings.")


def write_summary_csv(summary_rows: list, output_dir: Path):
    """Write the requested summary table."""
    grouped = defaultdict(list)
    counts_first = Counter()
    by_pair = defaultdict(list)

    for row in summary_rows:
        grouped[(row["distance"], row["representation"])].append(row)
        by_pair[(row["dataset"], row["architecture"])].append(row)

    for rows in by_pair.values():
        best_row = min(rows, key=lambda row: to_float(row, "mean_rank"))
        counts_first[(best_row["distance"], best_row["representation"])] += 1

    out_rows = []
    for representation in REPRESENTATION_ORDER:
        for distance in DISTANCE_ORDER:
            rows = grouped.get((distance, representation), [])
            if not rows:
                continue
            out_rows.append(
                {
                    "distance": distance,
                    "representation": representation,
                    "mean_mean_rank": mean_or_nan([to_float(row, "mean_rank") for row in rows]),
                    "mean_overall_rank": mean_or_nan([to_float(row, "overall_rank") for row in rows]),
                    "times_best_configuration": counts_first[(distance, representation)],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary_logits_distance.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "distance",
                "representation",
                "mean_mean_rank",
                "mean_overall_rank",
                "times_best_configuration",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[CSV] Saved summary table -> {output_path}")


def main():
    """Generate all requested figures and the summary table."""
    summary_path = _pick_existing_path(DEFAULT_SUMMARY_CSV, FALLBACK_SUMMARY_CSV)
    comparison_path = _pick_existing_path(DEFAULT_COMPARISON_CSV, FALLBACK_COMPARISON_CSV)
    output_dir = _pick_output_dir(DEFAULT_OUTPUT_DIR, FALLBACK_OUTPUT_DIR)

    summary_rows = load_csv_rows(summary_path)
    comparison_rows = load_csv_rows(comparison_path)

    print(f"[INPUT] Summary CSV: {summary_path}")
    print(f"[INPUT] Comparison CSV: {comparison_path}")
    print(f"[OUTPUT] Figures directory: {output_dir}")

    plot_heatmap_rank(
        summary_rows,
        value_key="mean_rank",
        title="Average mean rank by representation and distance",
        filename="heatmap_mean_rank",
        output_dir=output_dir,
    )
    plot_heatmap_rank(
        summary_rows,
        value_key="overall_rank",
        title="Average overall rank by representation and distance",
        filename="heatmap_overall_rank",
        output_dir=output_dir,
    )
    plot_slope_charts(summary_rows, output_dir)
    plot_logits_win_heatmap(comparison_rows, output_dir)
    plot_rank_distributions(summary_rows, output_dir)
    plot_best_configuration_counts(summary_rows, output_dir)
    plot_bump_chart(summary_rows, output_dir)
    write_summary_csv(summary_rows, output_dir)

    print("[DONE] All requested logits-vs-embeddings figures were generated.")


if __name__ == "__main__":
    main()
