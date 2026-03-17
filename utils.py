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

def plot_train_losses(n_epochs, losses, figsize=(12, 8), ls='.-') -> plt.Axes:

    msg = f"Mismatch between number of epochs ({n_epochs}) and losses length ({len(losses)})"
    assert n_epochs == len(losses), msg

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(range(n_epochs), losses, ls)
    ax.set_title("Training losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    return ax


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


def plot_wrong_mnist():
    return
