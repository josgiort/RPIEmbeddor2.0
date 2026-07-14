import numpy as np
import matplotlib.pyplot as plt


def window_weights(window_len):

    ramp_width = window_len // 4
    sigmoid_center = ramp_width // 2

    i = np.arange(1, window_len + 1, dtype=np.float32)
    w = np.ones(window_len, dtype=np.float32)

    ramp_up = i < ramp_width
    w[ramp_up] = 1.0 / (
        1.0 + np.exp(-(i[ramp_up] - sigmoid_center) / 16)
    )

    ramp_down = i >= (window_len - ramp_width)
    w[ramp_down] = 1.0 / (
        1.0 + np.exp((i[ramp_down] - window_len + sigmoid_center) / 16)
    )

    return w


window_len = 1022

weights = window_weights(window_len)

plt.figure(figsize=(6, 3))

plt.plot(
    np.arange(1, window_len + 1),
    weights,
    linewidth=2
)




plt.axvspan(1, 255, alpha=0.15)
plt.axvspan(255, 766, alpha=0.10)
plt.axvspan(766, 1022, alpha=0.15)

plt.text(100, 0.5, "Ramp-up", ha="center")
plt.text(510, 1.02, "Full weight", ha="center")
plt.text(900, 0.5, "Ramp-down", ha="center")

plt.xlabel("Position within window")
plt.ylabel("Weight")
plt.title("Sliding-window weighting function")

plt.ylim(-0.05, 1.05)

plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("sliding_window_weights.pdf")
plt.savefig("sliding_window_weights.png", dpi=300)

plt.show()


