import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Following line seemed to solve issue on MacOS due to probably conflict between gym environment and matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_scores(scores, rolling_window="auto", path: str = '.', threshold=None, prefix="", log=False):
    """Plot scores and optional rolling mean using specified window."""
    print(f"plot_scores, len(scores): {len(scores)}, scores[0]={scores[0]}")
    if len(scores) == 0:
        return

    if rolling_window == "auto":
        rolling_window = round(len(scores) / 4)

    plt.figure(figsize=(15, 8))
    if type(scores[0]) not in [float, int, np.float64, np.int64] and len(scores[0]) > 1:
        scores = np.array(scores)
        print(f"scores.shape={scores.shape}")
        nr, nc = scores.shape
        for d in range(scores.shape[0]):
            plt.subplot(nr, 1, d + 1)
            _plot_scores(scores[d, :], rolling_window, log)

    else:
        _plot_scores(scores, rolling_window, log)

    if threshold:
        plt.axhline(threshold, c="r", ls="--", label=f"Objective: {threshold}")

    plt.title("Scores")
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{path}/{prefix}_scores_{len(scores)}_episodes_{pd.Timestamp.utcnow().value}.png")


def _plot_scores(scores, rolling_window, log=False):
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    if log:
        rolling_mean = rolling_mean.abs()
        print(f"pd.Series(np.sign(scores)): {pd.Series(np.sign(scores))}")
        plt.semilogy(np.abs(scores), "c", label=f"sign: {int(pd.Series(np.sign(scores)).mode())}")
        plt.semilogy(rolling_mean, "k")
    else:
        plt.plot(scores, "c")
        plt.plot(rolling_mean, "k")
