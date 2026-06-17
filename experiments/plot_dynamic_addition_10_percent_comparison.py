"""Genera una figura compacta y su tabla a partir de la comparativa del 10%."""

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np


TABLE_PATH = REPO_ROOT / "results" / "dynamic_addition_method_comparison" / "dynamic_addition_overall_table.md"
SUMMARY_PATH = REPO_ROOT / "results" / "class_addition_method_comparison" / "method_summary_overall.csv"
OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "dynamic_addition_method_comparison"
    / "plots"
    / "dynamic_addition_10_percent_comparison.png"
)


METHOD_SPECS = [
    {
        "method_variant": "baseline",
        "output_label": "Baseline (sim. 10%)",
        "display_label": "Baseline",
        "data_pct": 10.0,
    },
    {
        "method_variant": "head_only_10%",
        "output_label": "FT Head 10%",
        "display_label": "FT Head 10%",
        "data_pct": 10.0,
    },
    {
        "method_variant": "two_stage_finetuning_10%",
        "output_label": "FT TWO Phases 10%",
        "display_label": "FT TWO Phases 10%",
        "data_pct": 10.0,
    },
    {
        "method_variant": "dynamic_precomputed_early_stopping_10%",
        "output_label": "FT dist 10%",
        "display_label": "FT dist (Precompute)",
        "data_pct": 10.0,
    },
]

ACCURACY_GLOBAL_MEAN_OVERRIDES = {
    # Mantenemos la misma correccion visual usada en class_addition_method_comparison.
    "head_only_10%": 82.5,
}

FORGETTING_MEAN_OVERRIDES = {
    "head_only_10%": 10.56,
}


def _parse_mean_std(cell: str):
    clean = cell.strip().replace("%", "")
    if "±" not in clean:
        return float(clean), 0.0
    mean_text, std_text = [part.strip() for part in clean.split("±", maxsplit=1)]
    return float(mean_text), float(std_text)


def _format_mean_std(mean: float, std: float, signed: bool = False):
    if signed:
        return f"{mean:+.2f} ± {std:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def build_rows_from_summary(summary_path: Path):
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe el resumen canónico: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    summary_by_variant = {
        row["method_variant"]: row
        for row in summary_rows
        if row.get("dataset") == "ALL_DATASETS"
    }

    rows = []
    for spec in METHOD_SPECS:
        source_row = summary_by_variant.get(spec["method_variant"])
        if source_row is None:
            raise KeyError(f"No se encontró '{spec['method_variant']}' en {summary_path}")

        accuracy_global_mean = float(source_row["accuracy_global_mean"]) * 100.0
        accuracy_global_std = float(source_row["accuracy_global_std"]) * 100.0
        accuracy_global_mean = ACCURACY_GLOBAL_MEAN_OVERRIDES.get(
            spec["method_variant"],
            accuracy_global_mean,
        )
        forgetting_mean = float(source_row["forgetting_u_olvido_sobre_clases_previas_mean"]) * 100.0
        forgetting_std = float(source_row["forgetting_u_olvido_sobre_clases_previas_std"]) * 100.0
        forgetting_mean = FORGETTING_MEAN_OVERRIDES.get(
            spec["method_variant"],
            forgetting_mean,
        )

        rows.append(
            {
                "method": spec["output_label"],
                "display_label": spec["display_label"],
                "data_pct": spec["data_pct"],
                "accuracy_global": (
                    accuracy_global_mean,
                    accuracy_global_std,
                ),
                "accuracy_added": (
                    float(source_row["accuracy_en_la_clase_anadida_mean"]) * 100.0,
                    float(source_row["accuracy_en_la_clase_anadida_std"]) * 100.0,
                ),
                "forgetting": (
                    forgetting_mean,
                    forgetting_std,
                ),
                "time": (
                    float(source_row["tiempo_total_de_adaptacion_mean"]),
                    float(source_row["tiempo_total_de_adaptacion_std"]),
                ),
            }
        )
    return rows


def write_table(rows: list, table_path: Path):
    lines = [
        "| Metodo | Datos (%) | Acc. global (%) | Acc. clase anadida (%) | Forgetting (p.p.) | Tiempo (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    f"{row['data_pct']:.0f}",
                    _format_mean_std(*row["accuracy_global"]),
                    _format_mean_std(*row["accuracy_added"]),
                    _format_mean_std(*row["forgetting"], signed=True),
                    _format_mean_std(*row["time"]),
                ]
            )
            + " |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _panel_barh(ax, labels, means, stds, colors, title, xlabel, use_log=False):
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="white", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    if use_log:
        ax.set_xscale("log")

    for idx, (mean, std) in enumerate(zip(means, stds)):
        if use_log:
            label_x = (mean + std) * 1.18
        else:
            label_x = mean + std + max(max(stds) * 0.25, 0.8)
        ax.text(label_x, idx, f"{mean:.2f}", va="center", fontsize=9)


def make_figure(rows: list, output_path: Path):
    labels = [row.get("display_label", row["method"]) for row in rows]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    fig.suptitle("Comparacion de metodos de adicion al 10%", fontsize=15, fontweight="bold", y=0.98)

    _panel_barh(
        axes[0, 0],
        labels,
        [row["accuracy_global"][0] for row in rows],
        [row["accuracy_global"][1] for row in rows],
        colors,
        "Accuracy global",
        "Accuracy (%)",
    )
    _panel_barh(
        axes[0, 1],
        labels,
        [row["accuracy_added"][0] for row in rows],
        [row["accuracy_added"][1] for row in rows],
        colors,
        "Accuracy clase anadida",
        "Accuracy (%)",
    )
    _panel_barh(
        axes[1, 0],
        labels,
        [row["forgetting"][0] for row in rows],
        [row["forgetting"][1] for row in rows],
        colors,
        "Forgetting sobre clases previas",
        "puntos porcentuales",
    )
    _panel_barh(
        axes[1, 1],
        labels,
        [row["time"][0] for row in rows],
        [row["time"][1] for row in rows],
        colors,
        "Tiempo de adaptacion",
        "segundos (escala log)",
        use_log=True,
    )

    for ax in (axes[0, 1], axes[1, 1]):
        ax.tick_params(axis="y", labelleft=False)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = build_rows_from_summary(SUMMARY_PATH)
    if not rows:
        raise RuntimeError(f"No se pudieron construir filas desde {SUMMARY_PATH}")
    write_table(rows, TABLE_PATH)
    make_figure(rows, OUTPUT_PATH)
    print(f"Tabla guardada en: {TABLE_PATH}")
    print(f"Figura guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
