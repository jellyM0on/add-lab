from src.model import Seq2SeqRNN
from src.vocab import encode_string, decode_indices
from src.dataset import format_input

def exact_match(model: Seq2SeqRNN, dataset: list[tuple[str, str]]) -> float:
    correct = sum(
        decode_indices(model.predict(encode_string(inp))) == tgt
        for inp, tgt in dataset
    )
    return correct / len(dataset)

def show_examples(model: Seq2SeqRNN, dataset: list[tuple[str, str]], n: int = 12) -> None:
    print("\nExamples:")
    for inp, tgt in dataset[:n]:
        pred = decode_indices(model.predict(encode_string(inp)))
        print(f"IN: {inp}   TARGET: {tgt}   PRED: {pred}")

# ============================================================
# Full evaluation (0+0 … 99+99)
# ============================================================

def evaluate_all_pairs(model: Seq2SeqRNN, norm_padding: bool = False) -> float:
    """Evaluate every pair from 0+0 to 99+99 and print a summary."""
    total, correct = 0, 0
    wrong_examples: list[tuple[str, str, str]] = []

    for a in range(100):
        for b in range(100):
            inp = format_input(a, b, norm_padding=norm_padding)
            tgt = str(a + b)
            pred = decode_indices(model.predict(encode_string(inp)))
            total += 1
            if pred == tgt:
                correct += 1
            elif len(wrong_examples) < 25:
                wrong_examples.append((inp, tgt, pred))

    acc = correct / total
    label = "00+00 to 99+99" if norm_padding else "0+0 to 99+99"
    print(f"\nFull {label} evaluation")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {acc:.4f}")

    if wrong_examples:
        print("\nSample mistakes:")
        for inp, tgt, pred in wrong_examples:
            print(f"  IN: {inp}   TARGET: {tgt}   PRED: {pred}")
    else:
        print("\nNo mistakes found.")

    return acc

def evaluate_all_pairs_by_length(model: Seq2SeqRNN, norm_padding: bool = False) -> None:
    stats: dict[int, dict[str, int]] = {
        1: {"correct": 0, "total": 0},
        2: {"correct": 0, "total": 0},
        3: {"correct": 0, "total": 0},
    }

    for a in range(100):
        for b in range(100):
            inp = format_input(a, b, norm_padding=norm_padding)
            tgt = str(a + b)
            pred = decode_indices(model.predict(encode_string(inp)))
            L = len(tgt)
            stats[L]["total"] += 1
            if pred == tgt:
                stats[L]["correct"] += 1

    print("\nAccuracy by target length:")
    for L in [1, 2, 3]:
        c = stats[L]["correct"]
        t = stats[L]["total"]
        acc = c / t if t > 0 else 0.0
        print(f"  Length {L}: {c}/{t} = {acc:.4f}")

# ============================================================
# EOS evaluations
# ============================================================

def evaluate_stopping(model: Seq2SeqRNN, max_len: int = 4, norm_padding: bool = False) -> None:
    hit_max = 0
    total = 0

    for a in range(100):
        for b in range(100):
            raw = model.predict(
                encode_string(format_input(a, b, norm_padding=norm_padding)),
                max_len=max_len,
            )
            total += 1
            if len(raw) == max_len:
                hit_max += 1

    print(f"\nStopping condition: {total - hit_max}/{total} predictions emitted EOS naturally")
    print(f"Hit max_len (no EOS): {hit_max}/{total}")

def evaluate_stopping_with_examples(
    model: Seq2SeqRNN,
    max_len: int = 4,
    n_examples: int = 20,
    norm_padding: bool = False,
) -> None:
    hit_max = 0
    total = 0
    no_eos_examples:    list[tuple[str, str, str]] = []
    early_eos_examples: list[tuple[str, str, str]] = []

    for a in range(100):
        for b in range(100):
            inp = format_input(a, b, norm_padding=norm_padding)
            tgt = str(a + b)
            raw  = model.predict(encode_string(inp), max_len=max_len)
            pred = decode_indices(raw)
            total += 1

            if len(raw) == max_len:
                hit_max += 1
                if len(no_eos_examples) < n_examples:
                    no_eos_examples.append((inp, tgt, pred))
            elif len(pred) < len(tgt):
                if len(early_eos_examples) < n_examples:
                    early_eos_examples.append((inp, tgt, pred))

    print(f"\nStopping condition: {total - hit_max}/{total} emitted EOS")
    print(f"Hit max_len (no EOS): {hit_max}/{total}")

    print("\n=== Examples: NO EOS (hit max_len) ===")
    for inp, tgt, pred in no_eos_examples:
        print(f"  IN: {inp} | TARGET: {tgt} | PRED: {pred}")

    print("\n=== Examples: EARLY EOS (too short) ===")
    for inp, tgt, pred in early_eos_examples:
        print(f"  IN: {inp} | TARGET: {tgt} | PRED: {pred}")