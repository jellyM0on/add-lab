"""
Usage:
    python train.py
    python train.py --hidden_size 256 --epochs 150 --train_size 8000 --val_size 1000 --test_size 1000
"""

import argparse
import random
import numpy as np

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
) -> tuple[Seq2SeqRNN, list[tuple[str, str]], list[tuple[str, str]], dict]:
    full_data = build_full_dataset(99)
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
    parser.add_argument("--hidden_input", type=str,   default="15+8",
                        help="Input string used for the hidden-state heatmap.")
    args = parser.parse_args()

    model, val_data, test_data, history = train_model(
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        patience=args.patience,
        lr=args.lr,
    )

    save_model(model, args.model_path)
    save_history(history, args.history_path)

    print("\nValidation examples:")
    show_examples(model, val_data, n=12)

    print("\nTest examples:")
    show_examples(model, test_data, n=12)

    plot_loss(history)
    plot_hidden_state_heatmap(model, args.hidden_input)

if __name__ == "__main__":
    main()