import numpy as np
import matplotlib.pyplot as plt
from src.model import Seq2SeqRNN
from src.vocab import encode_string

def plot_loss(history: dict, filename: str = "loss_vs_iteration.png") -> None:
    losses = history["loss"]
    epochs = np.arange(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved loss plot to {filename}")

def plot_hidden_state_heatmap(
    model: Seq2SeqRNN,
    input_str: str = "15+8",
    filename: str = "hidden_state_heatmap.png",
) -> None:
    input_indices = encode_string(input_str)
    _, enc_hs = model.encoder_forward(input_indices)

    H = np.hstack(enc_hs[1:]).T

    plt.figure(figsize=(10, 5))
    plt.imshow(H, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Hidden Unit")
    plt.ylabel("Time Step")
    plt.title(f"Encoder Hidden State Heatmap — input: '{input_str}'")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved hidden state heatmap to {filename}")