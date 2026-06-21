from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_loss(
    loss_history,
    ticker=None,
    save_path=None,
    show = True
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    epochs = range(
        1,
        len(loss_history) + 1
    )

    ax.plot(
        epochs,
        loss_history,
        linewidth=2,
        color="steelblue"
    )

    title = "Autoencoder Training Loss"

    if ticker is not None:
        title = f"{ticker} - {title}"

    ax.set_title(title)

    ax.set_xlabel("Epoch")

    ax.set_ylabel(
        "Reconstruction Loss (MSE)"
    )

    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()
        
    plt.close()
