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
    input_str: str = "15+08",
    filename: str = "hidden_state_heatmap.png",
) -> None:
    token_labels = list(input_str)

    input_indices = encode_string(input_str)
    _, enc_hs = model.encoder_forward(input_indices)

    H = np.hstack(enc_hs[1:])
    n_steps = H.shape[1]

    fig_width = max(6, n_steps * 0.9)
    plt.figure(figsize=(fig_width, 5))

    im = plt.imshow(H, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, label="Tanh activation")

    plt.xlabel("Time Step (input token)")
    plt.ylabel("Hidden Unit Index")
    plt.title(f"Encoder Hidden State Heatmap — input: '{input_str}'")

    if len(token_labels) == n_steps:
        plt.xticks(range(n_steps), token_labels, fontsize=11)
    else:
        print(
            f"Warning: token_labels length ({len(token_labels)}) "
            f"does not match number of steps ({n_steps}). Labels skipped."
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved hidden state heatmap to {filename}")