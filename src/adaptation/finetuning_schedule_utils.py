"""Utilidades compartidas para schedules de fine-tuning."""

import copy

from src.core.training import train_with_early_stopping


def set_all_parameters_trainable(model):
    """Descongela todos los parametros del modelo."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def resolve_finetuning_training_setup(
    *,
    two_stage_finetuning: bool,
    train_percentage: float,
    head_only_epochs: int,
    frozen_epochs: int,
    unfrozen_epochs: int,
):
    """Describe el regimen de entrenamiento seleccionado."""
    if two_stage_finetuning:
        return {
            "mode_label": "two_stage_finetuning",
            "mode_dirname": "two_stage_finetuning",
            "backbone_mode": "finetuned",
            "trainable_scope": "head_then_full_model",
            "train_percentage": float(train_percentage),
            "max_epochs": int(frozen_epochs + unfrozen_epochs),
            "head_epochs": int(frozen_epochs),
            "full_model_epochs": int(unfrozen_epochs),
            "description": (
                f"head-only for {frozen_epochs} epochs, then full-model "
                f"fine-tuning for {unfrozen_epochs} epochs"
            ),
        }

    return {
        "mode_label": "head_only",
        "mode_dirname": "head_only",
        "backbone_mode": "frozen",
        "trainable_scope": "head_only",
        "train_percentage": float(train_percentage),
        "max_epochs": int(head_only_epochs),
        "head_epochs": int(head_only_epochs),
        "full_model_epochs": 0,
        "description": f"head-only for {head_only_epochs} epochs",
    }


def run_finetuning_schedule(
    model,
    train_loader,
    val_loader,
    *,
    two_stage_finetuning: bool,
    head_only_epochs: int,
    frozen_epochs: int,
    unfrozen_epochs: int,
    lr: float,
    verbose: bool,
):
    """Ejecuta un fine-tuning solo-cabecera o en dos fases y fusiona el historial."""
    if not two_stage_finetuning:
        result = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            epochs=head_only_epochs,
            lr=lr,
            patience=None,
            checkpoint_path=None,
            verbose=verbose,
        )
        history = []
        for epoch_info in result["history"]:
            tagged_epoch = dict(epoch_info)
            tagged_epoch["phase"] = "head_only"
            history.append(tagged_epoch)
        result["history"] = history
        return result

    if verbose:
        print(f"    Stage 1/2: training head only for {frozen_epochs} epochs.")
    frozen_stage = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=frozen_epochs,
        lr=lr,
        patience=None,
        checkpoint_path=None,
        verbose=verbose,
    )
    frozen_stage_best_weights = copy.deepcopy(model.state_dict())

    set_all_parameters_trainable(model)

    if verbose:
        print(f"    Stage 2/2: fine-tuning full model for {unfrozen_epochs} epochs.")
    unfrozen_stage = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=unfrozen_epochs,
        lr=lr,
        patience=None,
        checkpoint_path=None,
        verbose=verbose,
    )

    combined_history = []
    for epoch_info in frozen_stage["history"]:
        tagged_epoch = dict(epoch_info)
        tagged_epoch["phase"] = "head_only"
        combined_history.append(tagged_epoch)

    for epoch_info in unfrozen_stage["history"]:
        tagged_epoch = dict(epoch_info)
        tagged_epoch["epoch"] = int(tagged_epoch["epoch"]) + len(frozen_stage["history"])
        tagged_epoch["phase"] = "full_model"
        combined_history.append(tagged_epoch)

    best_val_loss = frozen_stage["best_val_loss"]
    best_val_accuracy = frozen_stage["best_val_accuracy"]
    best_epoch = frozen_stage["best_epoch"]
    if unfrozen_stage["best_val_loss"] < best_val_loss:
        best_val_loss = unfrozen_stage["best_val_loss"]
        best_val_accuracy = unfrozen_stage["best_val_accuracy"]
        best_epoch = unfrozen_stage["best_epoch"] + len(frozen_stage["history"])
    elif frozen_stage_best_weights is not None:
        model.load_state_dict(frozen_stage_best_weights)

    return {
        "history": combined_history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "epochs_ran": len(combined_history),
    }
