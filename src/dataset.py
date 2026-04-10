import random
from src.vocab import encode_string, stoi, EOS

def format_input(a: int, b: int, norm_padding: bool = False) -> str:
    return f"{a:02d}+{b:02d}" if norm_padding else f"{a}+{b}"

def build_full_dataset(
    max_num: int = 99,
    norm_padding: bool = False,
) -> list[tuple[str, str]]:
    data = []
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            inp = format_input(a, b, norm_padding=norm_padding)
            tgt = str(a + b)
            data.append((inp, tgt))
    return data

def split_dataset(
    data: list[tuple[str, str]],
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    total_needed = train_size + val_size + test_size
    if total_needed > len(data):
        raise ValueError(
            f"Requested {total_needed} samples, but dataset only has {len(data)} samples."
        )

    shuffled = data[:]
    random.shuffle(shuffled)

    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:train_size + val_size]
    test_data = shuffled[train_size + val_size:train_size + val_size + test_size]

    return train_data, val_data, test_data

def prepare_pair(
    input_str: str,
    target_str: str,
) -> tuple[list[int], list[int]]:
    input_indices = encode_string(input_str)
    target_indices = encode_string(target_str) + [stoi[EOS]]
    return input_indices, target_indices