"""Comparacion entre variantes few-shot y metodos de adaptacion tras class removal."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from src.core.results_utils import load_json, save_json, write_csv


METHOD_FOLDERS = {
    "baseline": "class_removal_baseline",
    "frozen_backbone_head": "class_removal_frozen_backbone_head",
    "finetuning": "class_removal_finetuning",
    "prototypical_fewshot": "class_removal_prototypical_fewshot",
    "dynamic_precomputed": "dynamic_embedding_finetuning/precompute_embeddings_then_finetune",
    "dynamic_epoch1": "dynamic_embedding_finetuning/epoch1_embeddings_dynamic_finetune",
}


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _safe_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _extract_percentage_from_path(path_str: str):
    match = re.search(r"/porc_(\d+(?:\.\d+)?)", path_str)
    if not match:
        return None
    return float(match.group(1))


def _normalize_common_fields(payload: dict, source_name: str):
    summary = payload.get("summary", {})
    accuracy_global = (
        summary.get("accuracy_global")
        if isinstance(summary, dict) and summary.get("accuracy_global") is not None
        else payload.get("accuracy_global", payload.get("test_overall_accuracy"))
    )
    accuracy_remaining = (
        summary.get("accuracy_en_clases_restantes")
        if isinstance(summary, dict) and summary.get("accuracy_en_clases_restantes") is not None
        else payload.get("accuracy_en_clases_restantes", payload.get("test_mean_per_class_accuracy"))
    )
    total_time = (
        summary.get("tiempo_total_de_adaptacion")
        if isinstance(summary, dict) and summary.get("tiempo_total_de_adaptacion") is not None
        else payload.get("tiempo_total_de_adaptacion", payload.get("elapsed_seconds"))
    )
    examples_used = (
        summary.get("numero_de_ejemplos_utilizados")
        if isinstance(summary, dict) and summary.get("numero_de_ejemplos_utilizados") is not None
        else payload.get("numero_de_ejemplos_utilizados", payload.get("num_examples_used_for_adaptation"))
    )
    confidence = (
        summary.get("confianza_de_prediccion")
        if isinstance(summary, dict) and summary.get("confianza_de_prediccion") is not None
        else payload.get("confianza_de_prediccion", payload.get("prediction_confidence_mean"))
    )
    trainable_params = (
        summary.get("numero_de_parametros_entrenados_o_modificados")
        if isinstance(summary, dict) and summary.get("numero_de_parametros_entrenados_o_modificados") is not None
        else payload.get(
            "numero_de_parametros_entrenados_o_modificados",
            payload.get("num_trainable_parameters"),
        )
    )
    status = payload.get("status")
    if status is None and isinstance(summary, dict):
        status = summary.get("status")

    return {
        "source_name": source_name,
        "dataset": payload.get("dataset"),
        "model_name": payload.get("model_name"),
        "removed_class": payload.get("removed_class", payload.get("modified_class")),
        "status": status,
        "best_epoch": _safe_int(payload.get("best_epoch")),
        "epochs_ran": _safe_int(payload.get("epochs_ran")),
        "best_val_accuracy": _safe_float(payload.get("best_val_accuracy")),
        "tiempo_total_de_adaptacion": _safe_float(total_time),
        "accuracy_global": _safe_float(accuracy_global),
        "accuracy_en_clases_restantes": _safe_float(accuracy_remaining),
        "forgetting_u_olvido": _safe_float(payload.get("forgetting_u_olvido")),
        "numero_de_ejemplos_utilizados": _safe_int(examples_used),
        "confianza_de_prediccion": _safe_float(confidence),
        "numero_de_parametros_entrenados_o_modificados": _safe_int(trainable_params),
    }


def normalize_result_row(payload: dict, source_name: str):
    """Normaliza una ejecucion individual de cualquier metodo al mismo esquema."""
    row = _normalize_common_fields(payload, source_name)

    if source_name == "prototypical_fewshot":
        shots_per_class = _safe_int(payload.get("shots_per_class"))
        row.update(
            {
                "method_family": "fewshot",
                "method_variant": f"prototypical_fewshot_shots_{shots_per_class}",
                "shots_per_class": shots_per_class,
                "train_percentage": None,
                "training_mode": payload.get("method", "prototypical_fewshot"),
                "backbone_mode": payload.get("backbone_mode"),
                "trainable_scope": payload.get("trainable_scope"),
                "update_type": payload.get("update_type"),
            }
        )
        return row

    if source_name == "baseline":
        row.update(
            {
                "method_family": "adaptation",
                "method_variant": "baseline",
                "shots_per_class": None,
                "train_percentage": 100.0,
                "training_mode": "baseline",
                "backbone_mode": payload.get("backbone_mode"),
                "trainable_scope": payload.get("trainable_scope"),
                "update_type": "remove",
            }
        )
        return row

    if source_name in {"dynamic_precomputed", "dynamic_epoch1"}:
        metrics_path = str(payload.get("metrics_path", ""))
        train_percentage = _safe_float(payload.get("train_percentage"))
        if train_percentage is None:
            train_percentage = _extract_percentage_from_path(metrics_path)
        if train_percentage is None:
            return None
        method_name = payload.get("method", source_name)
        if "precompute_embeddings_then_finetune" in method_name:
            method_variant = f"dynamic_precomputed_{train_percentage:g}%"
        else:
            method_variant = f"dynamic_epoch1_{train_percentage:g}%"
        if "/early_stopping/" in metrics_path:
            method_variant = f"{method_variant}_early_stopping"

        row.update(
            {
                "method_family": "adaptation",
                "method_variant": method_variant,
                "shots_per_class": None,
                "train_percentage": train_percentage,
                "training_mode": method_name,
                "backbone_mode": payload.get("backbone_mode"),
                "trainable_scope": payload.get("trainable_scope"),
                "update_type": payload.get("update_type"),
                "dataset": payload.get("dataset"),
                "model_name": payload.get("model_name"),
                "removed_class": payload.get("modified_class"),
                "tiempo_total_de_adaptacion": _safe_float(
                    row.get("tiempo_total_de_adaptacion", payload.get("total_time"))
                ),
                "accuracy_global": _safe_float(
                    row.get("accuracy_global", payload.get("accuracy"))
                ),
                "accuracy_en_clases_restantes": _safe_float(
                    row.get("accuracy_en_clases_restantes", payload.get("mean_per_class_accuracy"))
                ),
                "forgetting_u_olvido": _safe_float(
                    row.get("forgetting_u_olvido", payload.get("forgetting_score"))
                ),
                "numero_de_ejemplos_utilizados": _safe_int(
                    row.get("numero_de_ejemplos_utilizados", payload.get("num_training_samples"))
                ),
                "numero_de_parametros_entrenados_o_modificados": _safe_int(
                    row.get(
                        "numero_de_parametros_entrenados_o_modificados",
                        payload.get("num_trainable_parameters"),
                    )
                ),
            }
        )
        return row

    train_percentage = _safe_float(payload.get("train_percentage"))
    training_mode = payload.get("training_mode", source_name)
    row.update(
        {
            "method_family": "adaptation",
            "method_variant": f"{training_mode}_{train_percentage:g}%" if train_percentage is not None else training_mode,
            "shots_per_class": None,
            "train_percentage": train_percentage,
            "training_mode": training_mode,
            "backbone_mode": payload.get("backbone_mode"),
            "trainable_scope": payload.get("trainable_scope"),
            "update_type": "remove",
        }
    )
    return row


def load_all_rows(results_root: Path):
    """Carga todas las ejecuciones disponibles para la comparacion."""
    rows = []
    for source_name, folder_name in METHOD_FOLDERS.items():
        method_dir = results_root / folder_name
        if not method_dir.exists():
            continue
        for metrics_path in sorted(method_dir.glob("**/final_metrics.json")):
            if any(part in {"plots", "percentage_summaries", "analysis", "method_comparison"} for part in metrics_path.parts):
                continue
            payload = load_json(metrics_path)
            payload["metrics_path"] = str(metrics_path)
            row = normalize_result_row(payload, source_name)
            if row is None:
                continue
            row["metrics_path"] = str(metrics_path)
            rows.append(row)
    return rows


def filter_completed_rows(rows: list):
    """Mantiene solo filas completadas o sin estado explicito."""
    filtered = []
    for row in rows:
        status = row.get("status")
        if status in (None, "completed", "skipped_existing"):
            filtered.append(row)
    return filtered


def _group_key(row: dict):
    return (
        row.get("dataset"),
        row.get("model_name"),
        row.get("removed_class"),
    )


def summarize_fewshot_variants(rows: list):
    """Resume las variantes few-shot por numero de shots."""
    grouped = {}
    for row in rows:
        if row.get("method_family") != "fewshot":
            continue
        key = row.get("shots_per_class")
        grouped.setdefault(key, []).append(row)

    summaries = []
    for shots_per_class, group_rows in sorted(grouped.items()):
        accuracies = [row["accuracy_global"] for row in group_rows if row.get("accuracy_global") is not None]
        times = [row["tiempo_total_de_adaptacion"] for row in group_rows if row.get("tiempo_total_de_adaptacion") is not None]
        summaries.append(
            {
                "shots_per_class": shots_per_class,
                "method_variant": f"prototypical_fewshot_shots_{shots_per_class}",
                "num_runs": len(group_rows),
                "accuracy_global_mean": None if not accuracies else float(np.mean(accuracies)),
                "accuracy_global_std": None if not accuracies else float(np.std(accuracies)),
                "tiempo_total_de_adaptacion_mean": None if not times else float(np.mean(times)),
                "tiempo_total_de_adaptacion_std": None if not times else float(np.std(times)),
            }
        )
    return summaries


def pick_best_fewshot_variant(fewshot_summaries: list):
    """Elige la variante few-shot con mayor accuracy media; desempata por menor tiempo."""
    if not fewshot_summaries:
        return None

    ranked = sorted(
        fewshot_summaries,
        key=lambda row: (
            -(row.get("accuracy_global_mean") if row.get("accuracy_global_mean") is not None else float("-inf")),
            row.get("tiempo_total_de_adaptacion_mean") if row.get("tiempo_total_de_adaptacion_mean") is not None else float("inf"),
            row.get("shots_per_class") if row.get("shots_per_class") is not None else float("inf"),
        ),
    )
    return ranked[0]


def build_best_fewshot_comparison(rows: list, best_fewshot_variant: dict):
    """Compara la mejor variante few-shot con todos los metodos de adaptacion disponibles."""
    if best_fewshot_variant is None:
        return []

    target_variant = best_fewshot_variant["method_variant"]
    best_fewshot_lookup = {}
    for row in rows:
        if row.get("method_variant") == target_variant:
            best_fewshot_lookup[_group_key(row)] = row

    comparison_rows = []
    for row in rows:
        if row.get("method_family") != "adaptation":
            continue
        match = best_fewshot_lookup.get(_group_key(row))
        if match is None:
            continue

        comparison_rows.append(
            {
                "dataset": row.get("dataset"),
                "model_name": row.get("model_name"),
                "removed_class": row.get("removed_class"),
                "comparison_method": row.get("method_variant"),
                "comparison_source_name": row.get("source_name"),
                "fewshot_method": target_variant,
                "fewshot_shots_per_class": match.get("shots_per_class"),
                "fewshot_accuracy_global": match.get("accuracy_global"),
                "fewshot_tiempo_total_de_adaptacion": match.get("tiempo_total_de_adaptacion"),
                "fewshot_numero_de_ejemplos_utilizados": match.get("numero_de_ejemplos_utilizados"),
                "adaptation_accuracy_global": row.get("accuracy_global"),
                "adaptation_tiempo_total_de_adaptacion": row.get("tiempo_total_de_adaptacion"),
                "adaptation_numero_de_ejemplos_utilizados": row.get("numero_de_ejemplos_utilizados"),
                "accuracy_delta_fewshot_minus_adaptation": (
                    None
                    if match.get("accuracy_global") is None or row.get("accuracy_global") is None
                    else float(match["accuracy_global"]) - float(row["accuracy_global"])
                ),
                "time_delta_fewshot_minus_adaptation": (
                    None
                    if match.get("tiempo_total_de_adaptacion") is None or row.get("tiempo_total_de_adaptacion") is None
                    else float(match["tiempo_total_de_adaptacion"]) - float(row["tiempo_total_de_adaptacion"])
                ),
            }
        )
    return comparison_rows


def summarize_best_fewshot_comparison(comparison_rows: list):
    """Agrega la comparacion del mejor few-shot frente a cada metodo rival."""
    grouped = {}
    for row in comparison_rows:
        key = (row.get("comparison_source_name"), row.get("comparison_method"))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (source_name, method_variant), group_rows in sorted(grouped.items()):
        fewshot_acc = [row["fewshot_accuracy_global"] for row in group_rows if row.get("fewshot_accuracy_global") is not None]
        fewshot_time = [row["fewshot_tiempo_total_de_adaptacion"] for row in group_rows if row.get("fewshot_tiempo_total_de_adaptacion") is not None]
        adaptation_acc = [row["adaptation_accuracy_global"] for row in group_rows if row.get("adaptation_accuracy_global") is not None]
        adaptation_time = [row["adaptation_tiempo_total_de_adaptacion"] for row in group_rows if row.get("adaptation_tiempo_total_de_adaptacion") is not None]
        acc_deltas = [row["accuracy_delta_fewshot_minus_adaptation"] for row in group_rows if row.get("accuracy_delta_fewshot_minus_adaptation") is not None]
        time_deltas = [row["time_delta_fewshot_minus_adaptation"] for row in group_rows if row.get("time_delta_fewshot_minus_adaptation") is not None]

        summaries.append(
            {
                "comparison_source_name": source_name,
                "comparison_method": method_variant,
                "num_matched_runs": len(group_rows),
                "fewshot_accuracy_global_mean": None if not fewshot_acc else float(np.mean(fewshot_acc)),
                "fewshot_tiempo_total_de_adaptacion_mean": None if not fewshot_time else float(np.mean(fewshot_time)),
                "adaptation_accuracy_global_mean": None if not adaptation_acc else float(np.mean(adaptation_acc)),
                "adaptation_tiempo_total_de_adaptacion_mean": None if not adaptation_time else float(np.mean(adaptation_time)),
                "accuracy_delta_fewshot_minus_adaptation_mean": None if not acc_deltas else float(np.mean(acc_deltas)),
                "time_delta_fewshot_minus_adaptation_mean": None if not time_deltas else float(np.mean(time_deltas)),
            }
        )
    return summaries


def build_markdown_report(best_fewshot_variant: dict, comparison_summary_rows: list):
    """Genera un markdown corto con la conclusion principal."""
    lines = [
        "# Comparacion few-shot vs adaptacion tras eliminacion",
        "",
    ]

    if best_fewshot_variant is None:
        lines.append("No se encontraron resultados few-shot completados.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Mejor variante few-shot",
            "",
            (
                f"- Variante ganadora: `{best_fewshot_variant['method_variant']}` "
                f"(shots={best_fewshot_variant['shots_per_class']})"
            ),
            (
                f"- Accuracy media: `{best_fewshot_variant['accuracy_global_mean']:.4f}`"
                if best_fewshot_variant.get("accuracy_global_mean") is not None
                else "- Accuracy media: `N/A`"
            ),
            (
                f"- Tiempo medio de adaptacion: `{best_fewshot_variant['tiempo_total_de_adaptacion_mean']:.2f}s`"
                if best_fewshot_variant.get("tiempo_total_de_adaptacion_mean") is not None
                else "- Tiempo medio de adaptacion: `N/A`"
            ),
            f"- Numero de ejecuciones usadas: `{best_fewshot_variant['num_runs']}`",
            "",
            "## Comparacion contra otros metodos",
            "",
            "| Metodo | Runs comparables | Accuracy few-shot | Accuracy rival | Delta accuracy | Tiempo few-shot | Tiempo rival | Delta tiempo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    ordered_rows = sorted(
        comparison_summary_rows,
        key=lambda row: (
            row.get("accuracy_delta_fewshot_minus_adaptation_mean") if row.get("accuracy_delta_fewshot_minus_adaptation_mean") is not None else float("-inf"),
            -(row.get("time_delta_fewshot_minus_adaptation_mean") if row.get("time_delta_fewshot_minus_adaptation_mean") is not None else float("inf")),
        ),
        reverse=True,
    )

    for row in ordered_rows:
        lines.append(
            "| {method} | {runs} | {f_acc} | {a_acc} | {d_acc} | {f_time} | {a_time} | {d_time} |".format(
                method=row["comparison_method"],
                runs=row["num_matched_runs"],
                f_acc=f"{row['fewshot_accuracy_global_mean']:.4f}" if row.get("fewshot_accuracy_global_mean") is not None else "N/A",
                a_acc=f"{row['adaptation_accuracy_global_mean']:.4f}" if row.get("adaptation_accuracy_global_mean") is not None else "N/A",
                d_acc=(
                    f"{row['accuracy_delta_fewshot_minus_adaptation_mean']:+.4f}"
                    if row.get("accuracy_delta_fewshot_minus_adaptation_mean") is not None
                    else "N/A"
                ),
                f_time=(
                    f"{row['fewshot_tiempo_total_de_adaptacion_mean']:.2f}s"
                    if row.get("fewshot_tiempo_total_de_adaptacion_mean") is not None
                    else "N/A"
                ),
                a_time=(
                    f"{row['adaptation_tiempo_total_de_adaptacion_mean']:.2f}s"
                    if row.get("adaptation_tiempo_total_de_adaptacion_mean") is not None
                    else "N/A"
                ),
                d_time=(
                    f"{row['time_delta_fewshot_minus_adaptation_mean']:+.2f}s"
                    if row.get("time_delta_fewshot_minus_adaptation_mean") is not None
                    else "N/A"
                ),
            )
        )

    return "\n".join(lines) + "\n"


def plot_fewshot_variant_summary(output_dir: Path, fewshot_summaries: list):
    """Grafica accuracy y tiempo de las variantes few-shot por shots."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "fewshot_variants_accuracy_time.png"

    ordered = sorted(fewshot_summaries, key=lambda row: row["shots_per_class"])
    labels = [f"{row['shots_per_class']} shots" for row in ordered]
    accuracies = [row.get("accuracy_global_mean") or 0.0 for row in ordered]
    times = [row.get("tiempo_total_de_adaptacion_mean") or 0.0 for row in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    axes[0].bar(range(len(labels)), accuracies, color=colors, edgecolor="white")
    axes[0].set_title("Accuracy media por variante few-shot", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(range(len(labels)), times, color=colors, edgecolor="white")
    axes[1].set_title("Tiempo medio por variante few-shot", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Segundos")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_best_fewshot_vs_adaptation(output_dir: Path, comparison_summary_rows: list):
    """Grafica la comparacion del mejor few-shot frente a los demas metodos."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "best_fewshot_vs_adaptation.png"

    ordered = sorted(
        comparison_summary_rows,
        key=lambda row: row.get("adaptation_accuracy_global_mean") if row.get("adaptation_accuracy_global_mean") is not None else float("-inf"),
        reverse=True,
    )
    labels = [row["comparison_method"] for row in ordered]
    fewshot_acc = [row.get("fewshot_accuracy_global_mean") or 0.0 for row in ordered]
    adaptation_acc = [row.get("adaptation_accuracy_global_mean") or 0.0 for row in ordered]
    fewshot_time = [row.get("fewshot_tiempo_total_de_adaptacion_mean") or 0.0 for row in ordered]
    adaptation_time = [row.get("adaptation_tiempo_total_de_adaptacion_mean") or 0.0 for row in ordered]

    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, fewshot_acc, width=width, label="Best few-shot", color="#4C956C")
    axes[0].bar(x + width / 2, adaptation_acc, width=width, label="Adaptation method", color="#D17B0F")
    axes[0].set_title("Accuracy media en runs comparables", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, fewshot_time, width=width, label="Best few-shot", color="#4C956C")
    axes[1].bar(x + width / 2, adaptation_time, width=width, label="Adaptation method", color="#D17B0F")
    axes[1].set_title("Tiempo medio de adaptacion", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Segundos")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_accuracy_vs_time_landscape(output_dir: Path, fewshot_summaries: list, comparison_summary_rows: list):
    """Grafica el paisaje global accuracy-tiempo para lectura rapida."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "accuracy_vs_time_landscape.png"

    def _normalize_plot_row(row):
        method = str(row.get("comparison_method") or "")

        if method == "baseline":
            return {
                "label": "Baseline",
                "group": "baseline",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method == "frozen_backbone_head":
            return {
                "label": "FT_Head 100%",
                "group": "ft_head",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method.startswith("head_only_"):
            percentage = method.removeprefix("head_only_")
            return {
                "label": f"FT_Head {percentage}",
                "group": "ft_head",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method == "two_stage_finetuning":
            return {
                "label": "FT_Two_Phase 100%",
                "group": "ft_two_phase",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method.startswith("two_stage_finetuning_"):
            percentage = method.removeprefix("two_stage_finetuning_")
            return {
                "label": f"FT_Two_Phase {percentage}",
                "group": "ft_two_phase",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method == "dynamic_precomputed_10%_early_stopping":
            return {
                "label": "FT_Select_Dist_Pre",
                "group": "dynamic_pre",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        if method == "dynamic_epoch1_10%_early_stopping":
            return {
                "label": "FT_Select_Dist_Epoch1",
                "group": "dynamic_epoch1",
                "accuracy": row.get("adaptation_accuracy_global_mean"),
                "time": row.get("adaptation_tiempo_total_de_adaptacion_mean"),
            }

        return None

    rows = []
    for row in fewshot_summaries:
        rows.append(
            {
                "label": f"{row['shots_per_class']}-shot",
                "group": "fewshot",
                "accuracy": row.get("accuracy_global_mean"),
                "time": row.get("tiempo_total_de_adaptacion_mean"),
            }
        )
    for row in comparison_summary_rows:
        normalized_row = _normalize_plot_row(row)
        if normalized_row is not None:
            rows.append(normalized_row)

    rows = [row for row in rows if row["accuracy"] is not None and row["time"] is not None]
    deduped_by_label = {}
    for row in rows:
        existing = deduped_by_label.get(row["label"])
        if existing is None:
            deduped_by_label[row["label"]] = row
            continue
        current_key = (float(row["accuracy"]), -float(row["time"]))
        existing_key = (float(existing["accuracy"]), -float(existing["time"]))
        if current_key > existing_key:
            deduped_by_label[row["label"]] = row
    rows = list(deduped_by_label.values())
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {
        "fewshot": "#1f77b4",
        "baseline": "#8c564b",
        "ft_head": "#ff7f0e",
        "ft_two_phase": "#2ca02c",
        "dynamic_pre": "#d62728",
        "dynamic_epoch1": "#d62728",
    }
    markers = {
        "fewshot": "o",
        "baseline": "X",
        "ft_head": "s",
        "ft_two_phase": "^",
        "dynamic_pre": "D",
        "dynamic_epoch1": "D",
    }
    sizes = {
        "fewshot": 160,
        "baseline": 170,
        "ft_head": 130,
        "ft_two_phase": 130,
        "dynamic_pre": 135,
        "dynamic_epoch1": 135,
    }

    for row in rows:
        ax.scatter(
            row["time"],
            row["accuracy"],
            s=sizes[row["group"]],
            color=colors[row["group"]],
            marker=markers[row["group"]],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

    placed_rows = sorted(rows, key=lambda item: (item["time"], item["accuracy"]))
    label_positions = []
    x_values = np.array([row["time"] for row in placed_rows], dtype=float)
    y_values = np.array([row["accuracy"] for row in placed_rows], dtype=float)
    x_span = max(float(np.max(x_values) - np.min(x_values)), 1e-9)
    y_span = max(float(np.max(y_values) - np.min(y_values)), 1e-9)
    x_threshold = max(x_span * 0.08, 1e-9)
    y_threshold = max(y_span * 0.06, 1e-9)
    offset_cycle = [
        (6, 6),
        (6, 18),
        (6, -10),
        (6, 30),
        (6, -22),
        (6, 42),
        (6, -34),
    ]

    for row in placed_rows:
        nearby_count = sum(
            1
            for other in label_positions
            if abs(row["time"] - other["time"]) <= x_threshold
            and abs(row["accuracy"] - other["accuracy"]) <= y_threshold
        )
        xytext = offset_cycle[min(nearby_count, len(offset_cycle) - 1)]
        ax.annotate(
            row["label"],
            (row["time"], row["accuracy"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
        )
        label_positions.append(
            {
                "time": row["time"],
                "accuracy": row["accuracy"],
            }
        )

    ax.set_xscale("log")
    ax.set_title("Accuracy vs tiempo de adaptacion", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tiempo de adaptacion (segundos, escala log)")
    ax.set_ylabel("Accuracy media")
    ax.grid(True, alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], marker=markers["fewshot"], color="w", label="Few-shot", markerfacecolor=colors["fewshot"], markersize=10),
        plt.Line2D([0], [0], marker=markers["baseline"], color="w", label="Baseline", markerfacecolor=colors["baseline"], markersize=10),
        plt.Line2D([0], [0], marker=markers["ft_head"], color="w", label="FT_Head", markerfacecolor=colors["ft_head"], markersize=9),
        plt.Line2D([0], [0], marker=markers["ft_two_phase"], color="w", label="FT_Two_Phase", markerfacecolor=colors["ft_two_phase"], markersize=9),
        plt.Line2D([0], [0], marker=markers["dynamic_pre"], color="w", label="FT_Select_Dist", markerfacecolor=colors["dynamic_pre"], markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_best_fewshot_deltas(output_dir: Path, comparison_summary_rows: list):
    """Grafica deltas del mejor few-shot respecto a cada metodo rival."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "best_fewshot_deltas.png"

    ordered = sorted(
        comparison_summary_rows,
        key=lambda row: row.get("accuracy_delta_fewshot_minus_adaptation_mean") if row.get("accuracy_delta_fewshot_minus_adaptation_mean") is not None else float("-inf"),
        reverse=True,
    )
    if not ordered:
        return None

    labels = [row["comparison_method"] for row in ordered]
    acc_deltas = [row.get("accuracy_delta_fewshot_minus_adaptation_mean") or 0.0 for row in ordered]
    time_deltas = [row.get("time_delta_fewshot_minus_adaptation_mean") or 0.0 for row in ordered]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    acc_colors = ["#4C956C" if value >= 0 else "#BC4749" for value in acc_deltas]
    axes[0].barh(y, acc_deltas, color=acc_colors, edgecolor="white")
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Delta accuracy: few-shot - rival", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Diferencia de accuracy")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].grid(axis="x", alpha=0.25)

    time_colors = ["#4C956C" if value <= 0 else "#BC4749" for value in time_deltas]
    axes[1].barh(y, time_deltas, color=time_colors, edgecolor="white")
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Delta tiempo: few-shot - rival", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Diferencia de tiempo (s)")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def save_comparison_outputs(output_dir: Path, all_rows: list, completed_rows: list):
    """Guarda todos los artefactos de la comparacion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_runs_normalized.csv", all_rows)
    write_csv(output_dir / "completed_runs_normalized.csv", completed_rows)

    fewshot_summaries = summarize_fewshot_variants(completed_rows)
    best_fewshot_variant = pick_best_fewshot_variant(fewshot_summaries)
    comparison_rows = build_best_fewshot_comparison(completed_rows, best_fewshot_variant)
    comparison_summary_rows = summarize_best_fewshot_comparison(comparison_rows)

    write_csv(output_dir / "fewshot_variant_summary.csv", fewshot_summaries)
    save_json(output_dir / "fewshot_variant_summary.json", fewshot_summaries)
    if best_fewshot_variant is not None:
        save_json(output_dir / "best_fewshot_variant.json", best_fewshot_variant)

    write_csv(output_dir / "best_fewshot_vs_adaptation_runs.csv", comparison_rows)
    save_json(output_dir / "best_fewshot_vs_adaptation_runs.json", comparison_rows)
    write_csv(output_dir / "best_fewshot_vs_adaptation_summary.csv", comparison_summary_rows)
    save_json(output_dir / "best_fewshot_vs_adaptation_summary.json", comparison_summary_rows)

    report_path = output_dir / "comparison_report.md"
    report_path.write_text(
        build_markdown_report(best_fewshot_variant, comparison_summary_rows),
        encoding="utf-8",
    )

    plot_fewshot_path = plot_fewshot_variant_summary(output_dir, fewshot_summaries) if fewshot_summaries else None
    plot_comparison_path = (
        plot_best_fewshot_vs_adaptation(output_dir, comparison_summary_rows)
        if comparison_summary_rows
        else None
    )
    plot_landscape_path = (
        plot_accuracy_vs_time_landscape(output_dir, fewshot_summaries, comparison_summary_rows)
        if fewshot_summaries or comparison_summary_rows
        else None
    )
    plot_deltas_path = (
        plot_best_fewshot_deltas(output_dir, comparison_summary_rows)
        if comparison_summary_rows
        else None
    )

    return {
        "fewshot_summaries": fewshot_summaries,
        "best_fewshot_variant": best_fewshot_variant,
        "comparison_rows": comparison_rows,
        "comparison_summary_rows": comparison_summary_rows,
        "report_path": report_path,
        "plot_fewshot_path": plot_fewshot_path,
        "plot_comparison_path": plot_comparison_path,
        "plot_landscape_path": plot_landscape_path,
        "plot_deltas_path": plot_deltas_path,
    }
