"""
Train PATE-GAN with Synthcity on tabular data with:
- automatic resume from latest checkpoint
- periodic checkpointing
- epoch-level CSV logging (generator/discriminator losses + privacy spent)
- final synthetic data export
- final model export

"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch



try:
    # Some Synthcity versions may expose this path.
    from synthcity.plugins.generic.plugin_pategan import PATEGANPlugin, Teachers
except ImportError:
    # Current Synthcity versions expose PATE-GAN under privacy plugins.
    from synthcity.plugins.privacy.plugin_pategan import PATEGANPlugin, Teachers

from synthcity.plugins.core.models.tabular_gan import TabularGAN



# User placeholders
TRAIN_DATA_PATH = "/content/drive/MyDrive/PATE-TransGAN/Adult/Adult_after/adult_train_preprocessed.csv"
OUTPUT_DIR = "/content/drive/MyDrive/PATE-GAN"

# Random seed to run index mapping used for automatic output sub-folders.
RUN_MAPPING = {
    42: 1,
    13: 2,
    101: 3,
    1234: 4,
    2026: 5,
}

# Core training hyperparameters
N_ITER = 200
BATCH_SIZE = 128
EPSILON = 1.0
N_TEACHERS = 100
TARGET_COLUMN = "salary"

# Checkpoint settings
CHECKPOINT_EVERY = 30

# Misc
RANDOM_STATE = 1234
DEVICE = "auto"  # "auto", "cpu", "cuda"
SYNTHETIC_ROWS = None  # None -> same as training set size


@dataclass
class RunState:
    plugin: Any
    current_iter: int
    epsilon_hat: float
    history: List[Dict[str, Any]]
    train_columns: List[str]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("DEVICE must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is set to 'cuda' but CUDA is not available.")
    return device


def resolve_run_name(random_state: int) -> str:
    run_number = RUN_MAPPING.get(int(random_state))
    if run_number is not None:
        return f"Run_{run_number}"
    return f"Run_seed_{int(random_state)}"


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logger = logging.getLogger("pategan_train")
    logger.setLevel(logging.INFO)

    # Close previous handlers to avoid file descriptor leaks when reinitializing per run.
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def setup_batch_logger() -> logging.Logger:
    logger = logging.getLogger("pategan_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_training_data(train_data_path: Path) -> pd.DataFrame:
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_data_path}")

    df = pd.read_csv(train_data_path)
    if df.empty:
        raise ValueError("Training CSV is empty.")
    return df


def checkpoint_path(output_dir: Path, iteration: int) -> Path:
    return output_dir / f"checkpoint_iter_{iteration:05d}.joblib"


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("checkpoint_iter_*.joblib"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def list_checkpoints_desc(output_dir: Path) -> List[Path]:
    return sorted(output_dir.glob("checkpoint_iter_*.joblib"), reverse=True)


def save_checkpoint(output_dir: Path, state: RunState) -> Path:
    ckpt = checkpoint_path(output_dir, state.current_iter)
    payload = {
        "plugin": state.plugin,
        "current_iter": state.current_iter,
        "epsilon_hat": state.epsilon_hat,
        "history": state.history,
        "train_columns": state.train_columns,
        "resumable": True,
    }

    try:
        joblib.dump(payload, ckpt)
    except Exception:
        # Some Colab/Synthcity builds contain dynamically generated classes
        # that cannot be pickled. Save a lightweight checkpoint instead.
        lightweight_payload = {
            "plugin": None,
            "current_iter": state.current_iter,
            "epsilon_hat": state.epsilon_hat,
            "history": state.history,
            "train_columns": state.train_columns,
            "resumable": False,
        }
        joblib.dump(lightweight_payload, ckpt)
    return ckpt


def load_checkpoint(checkpoint_file: Path) -> RunState:
    payload = joblib.load(checkpoint_file)
    if payload.get("plugin") is None:
        raise RuntimeError(
            "Checkpoint is metadata-only and cannot resume model state. "
            "Run with --force-restart to start a new training run."
        )

    return RunState(
        plugin=payload["plugin"],
        current_iter=int(payload["current_iter"]),
        epsilon_hat=float(payload["epsilon_hat"]),
        history=list(payload["history"]),
        train_columns=list(payload["train_columns"]),
    )


def build_plugin(
    n_iter: int,
    batch_size: int,
    epsilon: float,
    n_teachers: int,
    random_state: int,
    device: str,
    workspace: Path,
) -> Any:
    plugin = PATEGANPlugin(
        n_iter=n_iter,
        batch_size=batch_size,
        epsilon=epsilon,
        n_teachers=n_teachers,
        random_state=random_state,
        device=device,
        workspace=workspace,
    )
    return plugin


def initialize_manual_training_state(plugin: Any, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Reproduces the Synthcity PATEGAN.fit initialization so we can checkpoint/resume
    every outer iteration while preserving native behavior.
    """
    pategan = plugin.model

    pategan.columns = train_df.columns
    if pategan.delta is None:
        pategan.delta = 1 / (len(train_df) * np.sqrt(len(train_df)))

    pategan.model = TabularGAN(
        train_df,
        n_units_latent=pategan.generator_n_units_hidden,
        batch_size=pategan.batch_size,
        generator_n_layers_hidden=pategan.generator_n_layers_hidden,
        generator_n_units_hidden=pategan.generator_n_units_hidden,
        generator_nonlin=pategan.generator_nonlin,
        generator_nonlin_out_discrete="softmax",
        generator_nonlin_out_continuous="none",
        generator_lr=pategan.lr,
        generator_residual=True,
        generator_n_iter=pategan.generator_n_iter,
        generator_batch_norm=False,
        generator_dropout=0,
        generator_weight_decay=pategan.weight_decay,
        discriminator_n_units_hidden=pategan.discriminator_n_units_hidden,
        discriminator_n_layers_hidden=pategan.discriminator_n_layers_hidden,
        discriminator_n_iter=pategan.discriminator_n_iter,
        discriminator_nonlin=pategan.discriminator_nonlin,
        discriminator_batch_norm=False,
        discriminator_dropout=pategan.discriminator_dropout,
        discriminator_lr=pategan.lr,
        discriminator_weight_decay=pategan.weight_decay,
        clipping_value=pategan.clipping_value,
        encoder_max_clusters=pategan.encoder_max_clusters,
        encoder=pategan.encoder,
        n_iter_print=max(1, pategan.generator_n_iter - 1),
        device=pategan.device,
    )

    x_train_enc = pategan.model.encode(train_df)
    pategan.samples_per_teacher = max(1, int(len(x_train_enc) / max(1, pategan.n_teachers)))
    pategan.alpha_dict = np.zeros([pategan.alpha])

    epsilon_hat = 0.0
    return x_train_enc, epsilon_hat


def run_one_outer_iteration(
    plugin: Any,
    x_train_enc: pd.DataFrame,
    epsilon_hat: float,
) -> Tuple[float, float, float, int]:
    """
    Executes one outer PATE iteration:
    1) train teachers
    2) train student GAN
    3) update privacy accountant

    Returns:
        generator_loss_mean, discriminator_loss_mean, epsilon_hat, inner_steps
    """
    pategan = plugin.model

    teachers = Teachers(
        n_teachers=pategan.n_teachers,
        samples_per_teacher=pategan.samples_per_teacher,
        lamda=pategan.lamda,
        template=pategan.teacher_template,
    )
    teachers.fit(np.asarray(x_train_enc), pategan.model)

    def fake_labels_generator(x_batch: torch.Tensor) -> torch.Tensor:
        if pategan.n_teachers == 0:
            return torch.zeros((len(x_batch),))

        x_df = pd.DataFrame(x_batch.detach().cpu().numpy())
        n0_mb, n1_mb, y_mb = teachers.pate_lamda(np.asarray(x_df))

        if np.sum(y_mb) >= len(x_batch) / 2:
            return torch.zeros((len(x_batch),))

        pategan._update_alpha(n0_mb, n1_mb)
        return torch.from_numpy(np.reshape(np.asarray(y_mb, dtype=int), [-1, 1]))

    # Capture GAN epoch losses by wrapping internal train epoch.
    inner_losses: List[Tuple[float, float]] = []
    base_gan = pategan.model.model
    original_train_epoch = base_gan._train_epoch

    def wrapped_train_epoch(*args: Any, **kwargs: Any) -> Tuple[float, float]:
        g_loss, d_loss = original_train_epoch(*args, **kwargs)
        inner_losses.append((float(g_loss), float(d_loss)))
        return g_loss, d_loss

    base_gan._train_epoch = wrapped_train_epoch
    try:
        pategan.model.fit(
            x_train_enc,
            fake_labels_generator=fake_labels_generator,
            encoded=True,
        )
    finally:
        base_gan._train_epoch = original_train_epoch

    curr_list: List[float] = []
    for lidx in range(pategan.alpha):
        local_alpha = (pategan.alpha_dict[lidx] - np.log(pategan.delta)) / float(lidx + 1)
        curr_list.append(float(local_alpha))

    epsilon_hat = float(np.min(curr_list))

    if inner_losses:
        g_loss_mean = float(np.mean([x[0] for x in inner_losses]))
        d_loss_mean = float(np.mean([x[1] for x in inner_losses]))
    else:
        g_loss_mean = float("nan")
        d_loss_mean = float("nan")

    return g_loss_mean, d_loss_mean, epsilon_hat, len(inner_losses)


def write_history_csv(history: List[Dict[str, Any]], output_dir: Path) -> Path:
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history_path


def generate_synthetic_dataframe(plugin: Any, count: int) -> pd.DataFrame:
    syn: Any = None
    errors: List[str] = []

    # Try plugin API first.
    try:
        syn = plugin.generate(count=count)
    except Exception as e:
        errors.append(f"plugin.generate(count=...): {e}")

    # If plugin wrapper reports not fitted, call implementation methods directly.
    if syn is None and hasattr(plugin, "_generate"):
        try:
            syn = plugin._generate(count=count)
        except Exception as e:
            errors.append(f"plugin._generate(count=...): {e}")

    # Compatibility fallbacks across Synthcity internals.
    if syn is None and hasattr(plugin, "model"):
        pmodel = plugin.model

        if hasattr(pmodel, "generate"):
            try:
                syn = pmodel.generate(count=count)
            except Exception as e:
                errors.append(f"plugin.model.generate(count=...): {e}")

        if syn is None and hasattr(pmodel, "_generate"):
            try:
                syn = pmodel._generate(count=count)
            except Exception as e:
                errors.append(f"plugin.model._generate(count=...): {e}")

        # Nested GAN object used by some versions.
        if syn is None and hasattr(pmodel, "model"):
            inner = pmodel.model

            if hasattr(inner, "generate"):
                try:
                    syn = inner.generate(count=count)
                except Exception as e:
                    errors.append(f"plugin.model.model.generate(count=...): {e}")

            if syn is None and hasattr(inner, "sample"):
                try:
                    syn = inner.sample(count)
                except Exception as e:
                    errors.append(f"plugin.model.model.sample(count): {e}")

    if syn is None:
        raise RuntimeError(
            "Unable to generate synthetic data with available Synthcity APIs. "
            f"Tried methods and got: {' | '.join(errors)}"
        )

    if isinstance(syn, tuple) and len(syn) > 0:
        syn = syn[0]

    # Compatibility across synthcity versions:
    # - some return DataLoader-like objects with dataframe()
    # - some return DataFrame directly
    if hasattr(syn, "dataframe"):
        return syn.dataframe()
    if isinstance(syn, pd.DataFrame):
        return syn
    if isinstance(syn, np.ndarray):
        return pd.DataFrame(syn)
    if torch.is_tensor(syn):
        return pd.DataFrame(syn.detach().cpu().numpy())

    raise TypeError(
        "Unsupported output from synthetic generation. "
        f"Type received: {type(syn)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Synthcity PATE-GAN with checkpoints and logs")
    parser.add_argument("--train-data-path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--target-column", type=str, default=TARGET_COLUMN)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--n-teachers", type=int, default=N_TEACHERS)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--synthetic-rows", type=int, default=SYNTHETIC_ROWS)
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore available checkpoints and start training from scratch.",
    )
    # In notebooks (e.g., Colab), Jupyter injects args like "-f <kernel.json>".
    # parse_known_args keeps CLI behavior while safely ignoring those extras.
    args, _ = parser.parse_known_args()
    return args


def run_is_complete(output_dir: Path, target_epsilon: float, target_n_iter: int) -> bool:
    history_csv = output_dir / "training_history.csv"
    synthetic_csv = output_dir / "synthetic_data.csv"
    if not history_csv.exists() or not synthetic_csv.exists():
        return False

    try:
        history_df = pd.read_csv(history_csv)
        if history_df.empty:
            return False

        last_row = history_df.iloc[-1]
        last_epsilon = float(last_row.get("epsilon_hat", float("nan")))
        last_iter = int(last_row.get("iter", 0))
        return bool(last_epsilon >= target_epsilon or last_iter >= target_n_iter)
    except Exception:
        return False


def run_single_experiment(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    device: str,
    random_state: int,
) -> str:
    run_name = resolve_run_name(random_state)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("Starting run: %s", run_name)
    logger.info("Random state: %d", int(random_state))
    logger.info("Output directory: %s", output_dir)

    if not args.force_restart and run_is_complete(output_dir, float(args.epsilon), int(args.n_iter)):
        logger.info(
            "Run already completed (history + synthetic data found and stopping criterion met). "
            "Skipping this run."
        )
        return "skipped"

    latest_ckpt: Optional[Path] = None
    state: Optional[RunState] = None

    if not args.force_restart:
        for ckpt in list_checkpoints_desc(output_dir):
            logger.info("Trying checkpoint: %s", ckpt)
            try:
                state = load_checkpoint(ckpt)
                latest_ckpt = ckpt
                logger.info("Resuming from checkpoint: %s", latest_ckpt)
                break
            except Exception as e:
                logger.warning("Skipping checkpoint %s due to load error: %s", ckpt, e)

    if latest_ckpt is not None and state is not None:
        if state.train_columns != list(train_df.columns):
            raise ValueError(
                "Training columns differ from checkpoint columns. "
                "Cannot safely resume."
            )

        plugin = state.plugin
        current_iter = state.current_iter
        epsilon_hat = state.epsilon_hat
        history = state.history

        if not hasattr(plugin.model, "model"):
            raise RuntimeError(
                "Checkpoint does not include an initialized TabularGAN state. "
                "Please restart with --force-restart."
            )

        # Keep the latest command line budget as an upper bound.
        plugin.model.max_iter = int(args.n_iter)
        plugin.model.epsilon = float(args.epsilon)
        plugin.model.batch_size = int(args.batch_size)
        plugin.model.n_teachers = int(args.n_teachers)

        x_train_enc = plugin.model.model.encode(train_df)
        plugin.model.samples_per_teacher = max(
            1,
            int(len(x_train_enc) / max(1, plugin.model.n_teachers)),
        )
    else:
        logger.info("No checkpoint found. Starting a fresh training run.")
        plugin = build_plugin(
            n_iter=int(args.n_iter),
            batch_size=int(args.batch_size),
            epsilon=float(args.epsilon),
            n_teachers=int(args.n_teachers),
            random_state=int(random_state),
            device=device,
            workspace=output_dir / "workspace",
        )
        x_train_enc, epsilon_hat = initialize_manual_training_state(plugin, train_df)
        current_iter = 0
        history = []

    # Main outer training loop (privacy iterations)
    logger.info(
        "Starting training loop: target max_iter=%d, epsilon=%.6f, checkpoint_every=%d",
        int(args.n_iter),
        float(args.epsilon),
        int(args.checkpoint_every),
    )

    while epsilon_hat < float(args.epsilon) and current_iter < int(args.n_iter):
        current_iter += 1
        t0 = time.time()

        g_loss, d_loss, epsilon_hat, inner_steps = run_one_outer_iteration(
            plugin=plugin,
            x_train_enc=x_train_enc,
            epsilon_hat=epsilon_hat,
        )

        elapsed = time.time() - t0

        row = {
            "iter": current_iter,
            "generator_loss": g_loss,
            "discriminator_loss": d_loss,
            "epsilon_hat": epsilon_hat,
            "epsilon_target": float(args.epsilon),
            "delta": float(plugin.model.delta),
            "privacy_spent": epsilon_hat,
            "inner_gan_steps": inner_steps,
            "elapsed_seconds": elapsed,
        }
        history.append(row)

        logger.info(
            "Iter %d | G_loss=%.6f | D_loss=%.6f | epsilon_hat=%.6f/%.6f | inner_steps=%d | %.2fs",
            current_iter,
            g_loss,
            d_loss,
            epsilon_hat,
            float(args.epsilon),
            inner_steps,
            elapsed,
        )

        if current_iter % int(args.checkpoint_every) == 0:
            ckpt_file = save_checkpoint(
                output_dir,
                RunState(
                    plugin=plugin,
                    current_iter=current_iter,
                    epsilon_hat=epsilon_hat,
                    history=history,
                    train_columns=list(train_df.columns),
                ),
            )
            logger.info("Checkpoint saved: %s", ckpt_file)

    # Always save final checkpoint
    final_ckpt = save_checkpoint(
        output_dir,
        RunState(
            plugin=plugin,
            current_iter=current_iter,
            epsilon_hat=epsilon_hat,
            history=history,
            train_columns=list(train_df.columns),
        ),
    )
    logger.info("Final checkpoint saved: %s", final_ckpt)

    # Persist final model separately
    final_model_path = output_dir / "pategan_final_model.joblib"
    try:
        joblib.dump(plugin, final_model_path)
        logger.info("Final model saved: %s", final_model_path)
    except Exception as e:
        logger.warning(
            "Final model could not be pickled in this environment (%s). "
            "Training outputs are still saved (history CSV and synthetic data).",
            e,
        )

    # Persist history CSV
    history_csv_path = write_history_csv(history, output_dir)
    logger.info("Training history saved: %s", history_csv_path)

    # Manual training bypasses plugin.fit(), so mark wrapper as fitted
    # to keep generation pathways compatible across Synthcity versions.
    if hasattr(plugin, "fitted"):
        plugin.fitted = True

    # Generate synthetic data
    n_rows = len(train_df) if args.synthetic_rows is None else int(args.synthetic_rows)
    synthetic_df = generate_synthetic_dataframe(plugin, count=n_rows)

    # Ensure same columns ordering if available.
    if list(train_df.columns) == list(synthetic_df.columns):
        synthetic_df = synthetic_df[train_df.columns]

    synthetic_path = output_dir / "synthetic_data.csv"
    synthetic_df.to_csv(synthetic_path, index=False)
    logger.info("Synthetic data saved: %s | shape=%s", synthetic_path, synthetic_df.shape)
    logger.info("Run completed successfully: %s", run_name)
    return "completed"


def main() -> None:
    args = parse_args()

    if not args.train_data_path:
        raise ValueError("Please set TRAIN_DATA_PATH or pass --train-data-path.")
    if not args.output_dir:
        raise ValueError("Please set OUTPUT_DIR or pass --output-dir.")
    if int(args.n_teachers) < 0:
        raise ValueError("--n-teachers must be >= 0.")

    device = resolve_device(args.device)
    batch_logger = setup_batch_logger()

    batch_logger.info("Loading training data...")
    train_df = load_training_data(Path(args.train_data_path))

    if args.target_column:
        if args.target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{args.target_column}' was not found in the training data. "
                f"Available columns: {list(train_df.columns)}"
            )
    else:
        raise ValueError("--target-column cannot be empty.")

    batch_logger.info("Device: %s", device)
    batch_logger.info("Train shape: %s", train_df.shape)
    batch_logger.info("Target column: %s", args.target_column)
    batch_logger.info("Number of teachers: %d", int(args.n_teachers))
    batch_logger.info("Base output directory: %s", Path(args.output_dir))
    batch_logger.info("Starting sequential multi-run training for all mapped seeds.")

    seeds: List[int] = list(RUN_MAPPING.keys())
    completed = 0
    skipped = 0
    failed = 0

    for seed in seeds:
        run_name = resolve_run_name(seed)
        batch_logger.info("\n--- [%s] seed=%d ---", run_name, seed)
        try:
            status = run_single_experiment(
                args=args,
                train_df=train_df,
                device=device,
                random_state=int(seed),
            )

            if status == "completed":
                completed += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            batch_logger.error("Run failed for seed=%d (%s): %s", seed, run_name, e)
            batch_logger.error("Traceback:\n%s", traceback.format_exc())
            continue

    batch_logger.info(
        "\nAll runs processed. Summary -> completed=%d, skipped=%d, failed=%d, total=%d",
        completed,
        skipped,
        failed,
        len(seeds),
    )


if __name__ == "__main__":
    main()
