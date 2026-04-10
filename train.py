"""
Usage:
    python train.py
    python train.py --hidden_size 256 --epochs 150 --train_size 8000 --val_size 1000 --test_size 1000
    python train.py --norm_padding
"""

import argparse
import random
import numpy as np
from pathlib import Path

from src.model import Seq2SeqRNN
from src.dataset import build_full_dataset, split_dataset, prepare_pair
from src.evaluate import exact_match, show_examples
from src.save import copy_params, load_params, save_model, save_history
from src.visualize import plot_loss, plot_hidden_state_heatmap
from src.vocab import VOCAB_SIZE

np.random.seed(42)
random.seed(42)

def train_model(
    hidden_size: int = 128,
    epochs: int = 100,
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    patience: int = 12,
    lr: float = 0.01,
    norm_padding: bool = False,
) -> tuple[Seq2SeqRNN, list[tuple[str, str]], list[tuple[str, str]], dict]:
    full_data = build_full_dataset(99, norm_padding=norm_padding)
    train_data, val_data, test_data = split_dataset(
        full_data,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    model = Seq2SeqRNN(vocab_size=VOCAB_SIZE, hidden_size=hidden_size, lr=lr)

    best_acc = -1.0
    best_params = None
    best_epoch = 0
    epochs_without_improve = 0
    loss_history: list[float] = []
    acc_history: list[float] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)

        total_loss = 0.0
        for inp, tgt in train_data:
            input_indices, target_indices = prepare_pair(inp, tgt)
            total_loss += model.train_step(input_indices, target_indices)

        avg_loss = total_loss / len(train_data)
        val_acc = exact_match(model, val_data)

        loss_history.append(avg_loss)
        acc_history.append(val_acc)

        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_exact={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = copy_params(model)
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_params is not None:
        load_params(model, best_params)

    test_acc = exact_match(model, test_data)

    print(f"\nBest validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Test accuracy: {test_acc:.4f}")

    history = {
        "loss": loss_history,
        "val_exact": acc_history,
        "best_val_exact": best_acc,
        "test_exact": test_acc,
        "norm_padding": norm_padding,
    }

    return model, val_data, test_data, history


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the seq2seq addition model.")
    parser.add_argument("--hidden_size",  type=int,   default=128)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--train_size",   type=int,   default=8000)
    parser.add_argument("--val_size",     type=int,   default=1000)
    parser.add_argument("--test_size",    type=int,   default=1000)
    parser.add_argument("--patience",     type=int,   default=12)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--model_path",   type=str,   default="secret_accountant_model.npz")
    parser.add_argument("--history_path", type=str,   default="training_history.npz")
    parser.add_argument("--hidden_input", type=str,   default=None,
                        help="Input string used for the hidden-state heatmap.")
    parser.add_argument(
        "--norm_padding",
        action="store_true",
        help="Force all inputs to use zero-padded format like 00+00 and save outputs under norm-padding-outputs/.",
    )
    args = parser.parse_args()

    output_dir = Path("norm-padding-outputs") if args.norm_padding else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / Path(args.model_path).name
    history_path = output_dir / Path(args.history_path).name

    hidden_input = args.hidden_input
    if hidden_input is None:
        hidden_input = "15+08" if args.norm_padding else "15+8"

    model, val_data, test_data, history = train_model(
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        patience=args.patience,
        lr=args.lr,
        norm_padding=args.norm_padding,
    )

    save_model(model, str(model_path))
    save_history(history, str(history_path))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved history to: {history_path}")

    print("\nValidation examples:")
    show_examples(model, val_data, n=12)

    print("\nTest examples:")
    show_examples(model, test_data, n=12)

    plot_loss(history, filename=str(output_dir / "loss_vs_iteration.png"))
    plot_hidden_state_heatmap(
        model,
        hidden_input,
        filename=str(output_dir / "hidden_state_heatmap.png"),
    )

if __name__ == "__main__":
    main()