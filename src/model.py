import numpy as np
from src.vocab import (
    VOCAB_SIZE, SOS, EOS,
    stoi, one_hot, softmax, cross_entropy, clip_grads,
)

class Seq2SeqRNN:
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_size: int = 128, lr: float = 0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        scale = 0.05

        # Encoder weights
        self.W_xh_enc = np.random.randn(hidden_size, vocab_size) * scale
        self.W_hh_enc = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h_enc  = np.zeros((hidden_size, 1))

        # Decoder weights
        self.W_xh_dec = np.random.randn(hidden_size, vocab_size) * scale
        self.W_hh_dec = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h_dec  = np.zeros((hidden_size, 1))

        # Output projection
        self.W_hy = np.random.randn(vocab_size, hidden_size) * scale
        self.b_y  = np.zeros((vocab_size, 1))

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def encoder_forward(
        self, input_indices: list[int]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        xs, hs = [], []
        h = np.zeros((self.hidden_size, 1))
        hs.append(h.copy())

        for idx in input_indices:
            x = one_hot(idx, self.vocab_size)
            h = np.tanh(self.W_xh_enc @ x + self.W_hh_enc @ h + self.b_h_enc)
            xs.append(x)
            hs.append(h.copy())

        return xs, hs

    def decoder_forward(
        self, target_indices: list[int], context_h: np.ndarray
    ) -> tuple[float, list, list, list]:
        xs = []
        hs = [context_h.copy()]
        probs = []

        prev_idx = stoi[SOS]
        loss = 0.0

        for t in range(len(target_indices)):
            x = one_hot(prev_idx, self.vocab_size)
            h_prev = hs[-1]
            h = np.tanh(self.W_xh_dec @ x + self.W_hh_dec @ h_prev + self.b_h_dec)
            y = self.W_hy @ h + self.b_y
            p = softmax(y)

            xs.append(x)
            hs.append(h.copy())
            probs.append(p)

            loss += cross_entropy(p, target_indices[t])

            prev_idx = target_indices[t]

        return loss, xs, hs, probs

    def forward(
        self, input_indices: list[int], target_indices: list[int]
    ) -> tuple[float, dict]:
        enc_xs, enc_hs = self.encoder_forward(input_indices)
        context = enc_hs[-1]
        loss, dec_xs, dec_hs, dec_probs = self.decoder_forward(target_indices, context)

        cache = {
            "enc_xs": enc_xs,
            "enc_hs": enc_hs,
            "dec_xs": dec_xs,
            "dec_hs": dec_hs,
            "dec_probs": dec_probs,
            "target_indices": target_indices,
        }
        return loss, cache

    # ----------------------------------------------------------
    # Backward pass
    # ----------------------------------------------------------

    def backward(self, cache: dict) -> None:
        enc_xs        = cache["enc_xs"]
        enc_hs        = cache["enc_hs"]
        dec_xs        = cache["dec_xs"]
        dec_hs        = cache["dec_hs"]
        dec_probs     = cache["dec_probs"]
        target_indices = cache["target_indices"]

        dW_xh_enc = np.zeros_like(self.W_xh_enc)
        dW_hh_enc = np.zeros_like(self.W_hh_enc)
        db_h_enc  = np.zeros_like(self.b_h_enc)

        dW_xh_dec = np.zeros_like(self.W_xh_dec)
        dW_hh_dec = np.zeros_like(self.W_hh_dec)
        db_h_dec  = np.zeros_like(self.b_h_dec)

        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(target_indices))):
            p = dec_probs[t].copy()
            p[target_indices[t], 0] -= 1.0          # softmax–CE gradient

            dW_hy += p @ dec_hs[t + 1].T
            db_y  += p

            dh = self.W_hy.T @ p + dh_next
            h      = dec_hs[t + 1]
            h_prev = dec_hs[t]
            dtanh  = (1.0 - h * h) * dh

            dW_xh_dec += dtanh @ dec_xs[t].T
            dW_hh_dec += dtanh @ h_prev.T
            db_h_dec  += dtanh

            dh_next = self.W_hh_dec.T @ dtanh

        for t in reversed(range(len(enc_xs))):
            h      = enc_hs[t + 1]
            h_prev = enc_hs[t]
            dtanh  = (1.0 - h * h) * dh_next

            dW_xh_enc += dtanh @ enc_xs[t].T
            dW_hh_enc += dtanh @ h_prev.T
            db_h_enc  += dtanh

            dh_next = self.W_hh_enc.T @ dtanh

        grads = [
            dW_xh_enc, dW_hh_enc, db_h_enc,
            dW_xh_dec, dW_hh_dec, db_h_dec,
            dW_hy, db_y,
        ]
        clip_grads(grads, 1.0)

        self.W_xh_enc -= self.lr * dW_xh_enc
        self.W_hh_enc -= self.lr * dW_hh_enc
        self.b_h_enc  -= self.lr * db_h_enc

        self.W_xh_dec -= self.lr * dW_xh_dec
        self.W_hh_dec -= self.lr * dW_hh_dec
        self.b_h_dec  -= self.lr * db_h_dec

        self.W_hy -= self.lr * dW_hy
        self.b_y  -= self.lr * db_y

    def train_step(self, input_indices: list[int], target_indices: list[int]) -> float:
        loss, cache = self.forward(input_indices, target_indices)
        self.backward(cache)
        return float(loss)

    def predict(self, input_indices: list[int], max_len: int = 4) -> list[int]:
        _, enc_hs = self.encoder_forward(input_indices)
        h = enc_hs[-1]

        prev_idx = stoi[SOS]
        outputs = []

        for _ in range(max_len):
            x = one_hot(prev_idx, self.vocab_size)
            h = np.tanh(self.W_xh_dec @ x + self.W_hh_dec @ h + self.b_h_dec)
            y = self.W_hy @ h + self.b_y
            p = softmax(y)
            pred_idx = int(np.argmax(p))

            if pred_idx == stoi[EOS]:
                break

            outputs.append(pred_idx)
            prev_idx = pred_idx

        return outputs