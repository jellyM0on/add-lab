"""
Usage:
    python test.py --test
    python test.py --test_all
    python test.py --plots
    python test.py --test_all --plots
    python test.py --norm_padding --test_all
"""

import argparse
from pathlib import Path

from src.vocab import encode_string, decode_indices
from src.save import load_model, load_history
from src.evaluate import (
    evaluate_all_pairs,
    evaluate_all_pairs_by_length,
    evaluate_stopping,
    evaluate_stopping_with_examples,
)
from src.visualize import plot_loss, plot_hidden_state_heatmap

# ============================================================
# REPL
# ============================================================

def _validate_input(user_input: str, norm_padding: bool = False) -> str | None:
    """Return an error message if the input is invalid, else None."""
    if user_input.count("+") != 1:
        return "Input must contain exactly one '+' sign."
    if any(ch not in "0123456789+" for ch in user_input):
        return "Invalid characters. Use digits and '+' only."

    left, right = user_input.split("+")
    if not left or not right:
        return "Both sides of '+' must be non-empty."

    if norm_padding:
        if len(left) != 2 or len(right) != 2:
            return "With --norm_padding, input must be in exactly this format: 00+00."

    return None

def run_interactive(model, norm_padding: bool = False) -> None:
    if norm_padding:
        print("\nEnter expressions like 03+05, 11+07, 15+08 (type 'quit' to exit):")
    else:
        print("\nEnter expressions like 1+1, 2+13, 15+8 (type 'quit' to exit):")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "quit":
            break
        error = _validate_input(user_input, norm_padding=norm_padding)
        if error:
            print(error)
            continue
        pred = decode_indices(model.predict(encode_string(user_input)))
        print(f"Predicted: {pred}")

# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate or interact with a trained seq2seq addition model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --test
  python test.py --test_all
  python test.py --plots --hidden_input 42+58
  python test.py --norm_padding --test
  python test.py --norm_padding --test_all --plots
        """,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Launch an interactive REPL to query the model.",
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="Evaluate every pair from 0+0 to 99+99 and print accuracy stats.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Regenerate the loss curve and encoder hidden-state heatmap.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="secret_accountant_model.npz",
        help="Path to the saved model weights.",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default="training_history.npz",
        help="Path to the saved training history.",
    )
    parser.add_argument(
        "--hidden_input",
        type=str,
        default=None,
        help="Input string used for the hidden-state heatmap.",
    )
    parser.add_argument(
        "--norm_padding",
        action="store_true",
        help="Expect zero-padded inputs like 00+00 and load files from norm-padding-outputs/ by default.",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.test, args.test_all, args.plots]):
        parser.print_help()
        return

    base_dir = Path("norm-padding-outputs") if args.norm_padding else Path(".")
    model_path = base_dir / Path(args.model_path).name
    history_path = base_dir / Path(args.history_path).name

    hidden_input = args.hidden_input
    if hidden_input is None:
        hidden_input = "15+08" if args.norm_padding else "15+8"

    model = load_model(str(model_path))

    if args.test_all:
        evaluate_all_pairs(model, norm_padding=args.norm_padding)
        evaluate_all_pairs_by_length(model, norm_padding=args.norm_padding)
        evaluate_stopping(model, norm_padding=args.norm_padding)
        evaluate_stopping_with_examples(model, norm_padding=args.norm_padding)

    if args.plots:
        history = load_history(str(history_path))
        plot_loss(history, filename=str(base_dir / "loss_vs_iteration.png"))
        plot_hidden_state_heatmap(
            model,
            hidden_input,
            filename=str(base_dir / "hidden_state_heatmap.png"),
        )

    if args.test:
        run_interactive(model, norm_padding=args.norm_padding)

if __name__ == "__main__":
    main()