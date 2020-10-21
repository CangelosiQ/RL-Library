import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
# Following line seemed to solve issue on MacOS due to probably conflict between gym environment and matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_scores(scores, rolling_window=100, path: str = '.', threshold=None, prefix=""):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores, "c"); plt.title("Scores")
    plt.xlabel('Episode #')
    plt.ylabel('Score')

    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, "k")
    if threshold:
        plt.axhline(threshold, c="r", ls="--", label="Objective: 13")
    plt.legend()
    plt.savefig(f"{path}/{prefix}_scores_{len(scores)}_episodes_{pd.Timestamp.utcnow().value}.png")
    return rolling_mean
