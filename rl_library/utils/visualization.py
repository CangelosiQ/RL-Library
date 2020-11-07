import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# Following line seemed to solve issue on MacOS due to probably conflict between gym environment and matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_scores(scores, rolling_window=100, path: str = '.', threshold=None, prefix=""):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure(figsize=(15, 8))
    if len(scores[0])>1:
        scores = np.aray(scores)
        nr, nc = scores.shape
        for d in range(scores.shape[1]):
            plt.subplot(nc, 1, d+1)
            plt.plot(scores[:, d], "c")
            rolling_mean = pd.Series(scores[:, d]).rolling(rolling_window).mean()
            plt.plot(rolling_mean, "k")

    else:
        plt.plot(scores, "c")
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean, "k")
        if threshold:
            plt.axhline(threshold, c="r", ls="--", label=f"Objective: {threshold}")

    plt.title("Scores")
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{path}/{prefix}_scores_{len(scores)}_episodes_{pd.Timestamp.utcnow().value}.png")
    return rolling_mean
