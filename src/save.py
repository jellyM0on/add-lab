import numpy as np
from src.model import Seq2SeqRNN
from src.vocab import VOCAB_SIZE

def copy_params(model: Seq2SeqRNN) -> dict[str, np.ndarray]:
    return {
        "W_xh_enc": model.W_xh_enc.copy(),
        "W_hh_enc": model.W_hh_enc.copy(),
        "b_h_enc":  model.b_h_enc.copy(),
        "W_xh_dec": model.W_xh_dec.copy(),
        "W_hh_dec": model.W_hh_dec.copy(),
        "b_h_dec":  model.b_h_dec.copy(),
        "W_hy":     model.W_hy.copy(),
        "b_y":      model.b_y.copy(),
    }

def load_params(model: Seq2SeqRNN, params: dict[str, np.ndarray]) -> None:
    model.W_xh_enc = params["W_xh_enc"]
    model.W_hh_enc = params["W_hh_enc"]
    model.b_h_enc  = params["b_h_enc"]
    model.W_xh_dec = params["W_xh_dec"]
    model.W_hh_dec = params["W_hh_dec"]
    model.b_h_dec  = params["b_h_dec"]
    model.W_hy     = params["W_hy"]
    model.b_y      = params["b_y"]

# ============================================================
# Save
# ============================================================

def save_model(model: Seq2SeqRNN, filename: str = "secret_accountant_model.npz") -> None:
    np.savez(
        filename,
        W_xh_enc=model.W_xh_enc,
        W_hh_enc=model.W_hh_enc,
        b_h_enc=model.b_h_enc,
        W_xh_dec=model.W_xh_dec,
        W_hh_dec=model.W_hh_dec,
        b_h_dec=model.b_h_dec,
        W_hy=model.W_hy,
        b_y=model.b_y,
        hidden_size=np.array([model.hidden_size]),
    )
    print(f"Model saved to {filename}")

def load_model(filename: str = "secret_accountant_model.npz") -> Seq2SeqRNN:
    data = np.load(filename)
    hidden_size = int(data["hidden_size"][0]) if "hidden_size" in data else 128

    model = Seq2SeqRNN(vocab_size=VOCAB_SIZE, hidden_size=hidden_size, lr=0.01)
    model.W_xh_enc = data["W_xh_enc"]
    model.W_hh_enc = data["W_hh_enc"]
    model.b_h_enc  = data["b_h_enc"]
    model.W_xh_dec = data["W_xh_dec"]
    model.W_hh_dec = data["W_hh_dec"]
    model.b_h_dec  = data["b_h_dec"]
    model.W_hy     = data["W_hy"]
    model.b_y      = data["b_y"]

    print(f"Model loaded from {filename}")
    return model

def save_history(history: dict, filename: str = "training_history.npz") -> None:
    np.savez(
        filename,
        loss=np.array(history["loss"]),
        val_exact=np.array(history["val_exact"]),
    )
    print(f"History saved to {filename}")

def load_history(filename: str = "training_history.npz") -> dict:
    data = np.load(filename)
    return {
        "loss":      data["loss"].tolist(),
        "val_exact": data["val_exact"].tolist(),
    }