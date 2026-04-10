"""
Usage:
    python test.py --test   # REPL
    python test.py --test_all # full evaluation (0+0 … 99+99)
    python test.py --plots  # regenerate loss curve & heatmap
    python test.py --test_all --plots # combine flags freely
"""

import argparse
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

def _validate_input(user_input: str) -> str | None:
    """Return an error message if the input is invalid, else None."""
    if user_input.count("+") != 1:
        return "Input must contain exactly one '+' sign."
    if any(ch not in "0123456789+" for ch in user_input):
        return "Invalid characters. Use digits and '+' only."
    left, right = user_input.split("+")
    if not left or not right:
        return "Both sides of '+' must be non-empty."
    return None

def run_interactive(model) -> None:
    print("\nEnter expressions like 1+1, 2+13, 15+8 (type 'quit' to exit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "quit":
            break
        error = _validate_input(user_input)
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
  python test.py --test_all --plots
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
        help="Path to the saved model weights (default: secret_accountant_model.npz).",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default="training_history.npz",
        help="Path to the saved training history (default: training_history.npz).",
    )
    parser.add_argument(
        "--hidden_input",
        type=str,
        default="15+8",
        help="Input string used for the hidden-state heatmap (default: '15+8').",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.test, args.test_all, args.plots]):
        parser.print_help()
        return

    model = load_model(args.model_path)

    if args.test_all:
        evaluate_all_pairs(model)
        evaluate_all_pairs_by_length(model)
        evaluate_stopping(model)
        evaluate_stopping_with_examples(model)

    if args.plots:
        history = load_history(args.history_path)
        plot_loss(history)
        plot_hidden_state_heatmap(model, args.hidden_input)

    if args.test:
        run_interactive(model)

if __name__ == "__main__":
    main()