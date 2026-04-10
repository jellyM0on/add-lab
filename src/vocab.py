import numpy as np

# ============================================================
# Vocabulary
# ============================================================

PAD = "_"
SOS = "<SOS>"
EOS = "<EOS>"

VOCAB = list("0123456789+") + [PAD, SOS, EOS]
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(VOCAB)

# ============================================================
# Encoding / decoding
# ============================================================

def encode_string(s: str) -> list[int]:
    return [stoi[ch] for ch in s]

def decode_indices(indices: list[int]) -> str:
    return "".join(itos[i] for i in indices if itos[i] not in (PAD, SOS, EOS))

# ============================================================
# Math helpers
# ============================================================

def one_hot(idx: int, size: int = VOCAB_SIZE) -> np.ndarray:
    v = np.zeros((size, 1))
    v[idx, 0] = 1.0
    return v

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def cross_entropy(probs: np.ndarray, target_idx: int) -> float:
    return -np.log(probs[target_idx, 0] + 1e-12)

def clip_grads(grads: list[np.ndarray], clip_value: float = 1.0) -> None:
    for g in grads:
        np.clip(g, -clip_value, clip_value, out=g)