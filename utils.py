# General purpose functions

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def print_params(params) -> None:
    """TODO: recursive"""

    for k in params.keys():
        print(f"{k}: {params[k]}")

    return


#----------------
# Plotting

def imshow(arr, cmap=plt.cm.binary, figsize=(2, 2)) -> None:
    """Wrapper for plt.imshow"""

    _, ax = plt.subplots(1, 1, figsize=(2, 2))
    if len(arr.shape) == 2:  # monochromatic
        ax.imshow(arr, cmap=cmap)
    elif len(arr.shape) == 3:  # RGB channels
        ax.imshow(arr)
    else:
        print("Invalid shape")
        return

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    return


def plot_train_losses(n_epochs, losses, figsize=(12, 8), ls='.-') -> plt.Axes:

    msg = f"Mismatch between number of epochs ({n_epochs}) and losses length ({len(losses)})"
    assert n_epochs == len(losses), msg

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(range(n_epochs), losses, ls)
    ax.set_title("Training losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    return ax


def plot_wrong_mnist(test_images, pred, real, neqs, img_width, img_height, nrows=12, ncols=10, figsize=(12, 12)):
    """Plot misclassified MNIST digits as a grid of images.

    Args:
        test_images (np.ndarray): Array of test images. Shape (N, 784) or (N, 28, 28).
        pred (np.ndarray): Predicted labels for all test images. Shape (N,).
        real (np.ndarray): True labels for all test images. Shape (N,).
        neqs (np.ndarray): Indices of misclassified examples where pred != real.
        img_width (int): Width of each image (pixels), typically 28.
        img_height (int): Height of each image (pixels), typically 28.
        nrows (int, optional): Number of rows in the output plot grid. Default is 12.
        ncols (int, optional): Number of columns in the output plot grid. Default is 10.
        figsize (tuple, optional): Matplotlib figure size. Default is (12, 12).

    Returns:
        matplotlib.figure.Figure: Figure object containing the plotted misclassified images.

    Notes:
        - The function visualizes the first `nrows * ncols` misclassified instances.
        - `test_images` is reshaped to (28, 28) for display.
    """

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        idx = neqs[i]
        ax.set_title(fr"{pred[idx]} $\neq$ {real[idx]}")
        digit = test_images[idx]
        if len(digit.shape) == 2:
            digit = digit.reshape(img_width, img_height)
            ax.imshow(digit, cmap="Blues_r")  #plt.cm.binary) #
        elif len(digit.shape) == 3:
            return  #TODO
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(fr"First {nrows * ncols} wrongly labeled instances (predicted $\neq$ real)", fontsize=16)
    fig.tight_layout()

    return fig


def plot_stock_prediction(y_test, y_test_pred, df, test_rmse, ticker_symbol="", figsize=(12, 8)) -> Figure:

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(4, 1)

    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(df.iloc[-len(y_test):].index, y_test, color='blue', label="Actual Price")
    ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color='green', label="Predicted Price")
    ax1.legend()
    plt.title(f"{ticker_symbol} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline(test_rmse, color='blue', ls='--', label="RMSE")
    ax2.plot(df.iloc[-len(y_test):].index, abs(y_test - y_test_pred), color='red', label="Prediction Error")
    ax2.legend()
    plt.title("Prediction Error")
    plt.xlabel("Date")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.show()

    return fig
