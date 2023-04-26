import matplotlib.pyplot as plt
import parameters as param
import numpy as np
import keras

def plot_hist_y_distribution(y_train, y_test, bins=100):
    # Plot the histogram for the train and test set y values

    hist_train, bins_train = np.histogram(y_train, bins=bins)
    hist_test, bins_test = np.histogram(y_test, bins=bins)

    plt.bar(bins_train[:-1], hist_train, width=(bins_train[1] - bins_train[0]), label="Train Distribution")
    plt.bar(bins_test[:-1], hist_test, width=(bins_test[1] - bins_test[0]), label="Test Distribution")
    plt.legend()
    plt.savefig(param.result_folder + "/Train and Test y Distribution.png")
    plt.show()

def plot_train_validation_loss(history: keras.callbacks.History, save_results = True):
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Improvement of the Network')
    plt.legend()
    if save_results:
        plt.savefig(param.result_folder + '/Network train and validation loss.png')
    plt.show()

def plot_history_metrics(history: keras.callbacks.History, save_results = True):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    if save_results:
        plt.savefig(param.result_folder + '/Network history metrics.png')
    plt.show()



def plot_scatter_true_vs_predicted(y_test, y_pred, start_:int, end_:int ,save_results = True):
    # start_ and end_ Specify the range you want to display

    # Plot the limited range of true values vs the predicted values
    plt.scatter(np.arange(start_, end_), y_pred[start_:end_], alpha=0.5, marker='x', color='red', label='Predicted')
    plt.scatter(np.arange(start_, end_), y_test.reshape(-1,1)[start_:end_], alpha=0.5, marker='o', color='blue', label='True')

    # plt.xlabel("True Values")
    plt.ylabel("Predicted/True Values")
    plt.title("True Values vs Predicted Values")
    plt.legend()
    if save_results:
        plt.savefig(param.result_folder + f'/True vs Predicted Values chunk: {param.chunk_size} - ma: {param.ma_len} - forecast: {param.forecast_size}.png')
    plt.show()

def plot_histogram_y_test_minus_y_pred(y_test, y_pred, save_results = True, bins = 30):
    # Calculate the differences between true and predicted values
    differences = (y_test - y_pred).flatten()

    # Plot the histogram of differences
    plt.hist(differences, bins=bins, color='purple')
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.title("Histogram of Differences between True and Predicted Values")
    if save_results:
        plt.savefig(param.result_folder + f'/Histogram of Differences between True and Predicted Values chunk: {param.chunk_size} - ma: {param.ma_len} - forecast: {param.forecast_size}.png')
    plt.show()


def plot_scatter_true_vs_predicted_diagonal(y_test, y_pred, save_results = True):
    # Plot the true values vs the predicted values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True Values vs Predicted Values\nChunk Len:{param.chunk_size} - SMA{param.ma_len} - Future Len:{param.forecast_size}")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') # Diagonal line
    if save_results:
        plt.savefig(param.result_folder + f'/True vs Predicted Values chunk: {param.chunk_size} - ma: {param.ma_len} - forecast: {param.forecast_size}.png')
    plt.show()