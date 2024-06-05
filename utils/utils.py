import matplotlib.pyplot as plt


def plot_training_history(history, metric="accuracy", save_img=True):
    # Plot training history
    plt.title(f"{metric} plot")
    plt.plot(history.history[metric])
    plt.plot(history.history[f"val_{metric}"])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    # plt.show()
    if save_img:
        plt.savefig(f"model/{metric}_plot.png", dpi=1000)
