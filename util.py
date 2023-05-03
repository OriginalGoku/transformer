import pandas as pd
import numpy as np
import random
import parameters as param
import matplotlib.pyplot as plt
import keras

# Code description:
#The gen_sliding_window function generates the sliding windows for a given dataset and returns the normalized features (X), labels (y), mean, and standard deviation values. This function is called separately for both the train and test datasets.
# The gen_multiple_sliding_window function processes multiple files and creates sliding windows for each file in the file_list. It separates the train and test datasets based on the train_cut_off_date.
# The gen_sliding_window function is called for both the train and test datasets to generate the sliding windows and normalize the data. The mean and standard deviation values are calculated based on the training data only, which prevents any information leakage.
# The gen_multiple_sliding_window function then returns the combined train and test datasets along with their respective mean and standard deviation values.

# def convert_z_normalized_data(X, y, mean_data, std_data):
#     original_X = [X[i] * std_data[i] + mean_data[i] for i in range(len(X))]
#     original_y = [y[i] * std_data[i] + mean_data[i] for i in range(len(y))]
#     return np.array(original_X), np.array(original_y)

def convert_to_original(normalized_window, mean_data, std_data, x_0=None):
    if x_0 is None:  # z_normalize = True
        # original_window = np.round(normalized_window * std_data + mean_data, 4)
        original_window = [normalized_window[i] * std_data[i] + mean_data[i] for i in range(len(normalized_window))]

    else:  # z_normalize = False
        # original_window = np.round((normalized_window / 100 + 1) * x_0, 4)
        original_window = [(normalized_window[i]+1)*x_0[i] for i in range(len(normalized_window))]

    return np.array(original_window)


# def gen_sliding_window(data, window_size, z_normalize):
#     print(f"Generating Sliding Window (window_size = {window_size}, z_normalize = {z_normalize})")
#     sliding_window = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size,))
#     n_windows = len(sliding_window)
#
#     sliding, sliding_y, mean_data, std_data, x_0 = [], [], [], [], []
#
#     for i in range(n_windows):
#         window = sliding_window[i]
#         if z_normalize:
#             mean = np.mean(window[:-1])
#             std = np.std(window[:-1])
#             mean_data.append(mean)
#             std_data.append(std)
#             normalized_window = (window - mean) / std
#         else:
#             normalized_window = (window / window[0]) - 1
#             x_0.append(window[0])
#
#         sliding.append(normalized_window[:-1])
#         sliding_y.append(normalized_window[-1])
#
#     return sliding, sliding_y, mean_data, std_data, x_0

def gen_sliding_window_including_open(data, open_data, window_size, z_normalize):
    print(f"Generating Sliding Window (window_size = {window_size}, z_normalize = {z_normalize})")
    sliding_window = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size,))
    open_data = open_data[window_size:]
    n_windows = len(sliding_window)-1

    sliding, sliding_y, mean_data, std_data, x_0 = [], [], [], [], []

    for i in range(n_windows):
        window = sliding_window[i]
        window = np.insert(window, -1, open_data[i])
        if z_normalize:
            mean = np.mean(window[:-1])
            std = np.std(window[:-1])
            mean_data.append(mean)
            std_data.append(std)
            normalized_window = (window - mean) / std
        else:
            normalized_window = (window / window[0]) - 1
            x_0.append(window[0])

        sliding.append(normalized_window[:-1])
        sliding_y.append(normalized_window[-1])

    return sliding, sliding_y, mean_data, std_data, x_0
def gen_multiple_sliding_window(file_list, window_size, z_normalize, train_cut_off_date, col_name):
    X_train, y_train, X_test, y_test, train_mean, train_std, test_mean, test_std, symbol_file, x_0_train, x_0_test = [], [], [], [], [], [], [], [], [], [], []
    for symbol_file in file_list:
        print(f"Processing {symbol_file}")
        data = load_file(symbol_file)
        data_train = data[data.index < train_cut_off_date]
        data_test = data[data.index >= train_cut_off_date]

        # X_train_temp, y_train_temp, train_mean, train_std, x_0_train_temp = gen_sliding_window(data_train[col_name], window_size, z_normalize)
        # X_test_temp, y_test_temp, test_mean, test_std, x_0_test_temp = gen_sliding_window(data_test[col_name], window_size, z_normalize)
        X_train_temp, y_train_temp, train_mean, train_std, x_0_train_temp = gen_sliding_window_including_open(data_train[col_name], data_train['OPEN'],
                                                                                               window_size, z_normalize)
        X_test_temp, y_test_temp, test_mean, test_std, x_0_test_temp = gen_sliding_window_including_open(data_test[col_name], data_test['OPEN'],
                                                                                          window_size, z_normalize)

        X_train.extend(X_train_temp)
        y_train.extend(y_train_temp)
        X_test.extend(X_test_temp)
        y_test.extend(y_test_temp)

        train_mean.extend(train_mean)
        train_std.extend(train_std)
        test_mean.extend(test_mean)
        test_std.extend(test_std)

        x_0_train.extend(x_0_train_temp)
        x_0_test.extend(x_0_test_temp)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(train_mean), np.array(train_std), np.array(test_mean), np.array(test_std), np.array(x_0_train), np.array(x_0_test)

def load_model(model_name: str):
    model = keras.models.load_model(model_name)
    return model


# Original
# def analyze_results(y_test, y_pred, X_test, save_results=True):
#     df_analyze_results = pd.DataFrame()
#     df_analyze_results['original'] = y_test.flatten()
#     df_analyze_results['predict'] = y_pred
#     df_analyze_results['direction'] = np.sign(
#         df_analyze_results['original'] * df_analyze_results['predict'])  # .astype(bool)
#     df_analyze_results['diff'] = df_analyze_results['original'] - df_analyze_results['predict']
#     df_analyze_results.dropna(inplace=True)
#
#     print("Mistake: %",
#           round(100 * len(df_analyze_results[df_analyze_results['direction'] == -1]) / len(df_analyze_results), 2))
#     print("Mistake more than +-1%: ", round(100 * len(df_analyze_results[(df_analyze_results['direction'] == -1) & (
#             (df_analyze_results['diff'] > 0.01) | (df_analyze_results['diff'] < -0.01))]) / len(df_analyze_results),
#                                             2))
#     if save_results:
#         save_csv(df_analyze_results, param.result_folder + "/"
#                                                            f'Analysis for chunk {param.chunk_size} - MA {param.ma_len} - Forecast {param.forecast_size}.csv')

def analyze_results(y_test, y_pred, X_test, save_results=True):
    df_analyze_results = pd.DataFrame()
    df_analyze_results['original'] = y_test.flatten()
    df_analyze_results['predict'] = y_pred
    df_analyze_results['last_x'] = X_test[:, -1]
    df_analyze_results['direction'] = [False] * len(df_analyze_results)
    df_analyze_results.loc[((df_analyze_results['original'] > df_analyze_results['last_x']) & (df_analyze_results['predict']>df_analyze_results['last_x']))|
                           ((df_analyze_results['original'] < df_analyze_results['last_x']) & (
                                       df_analyze_results['predict'] < df_analyze_results['last_x']))
    ,'direction'] = True
    # df_analyze_results['direction'] = np.sign(
    #     df_analyze_results['original'] * df_analyze_results['predict'])  # .astype(bool)
    # df_analyze_results['diff'] = df_analyze_results['original'] - df_analyze_results['predict']
    df_analyze_results['diff'] = -np.abs(df_analyze_results['predict'] - df_analyze_results['last_x'])/df_analyze_results['last_x']
    df_analyze_results.loc[df_analyze_results['direction']==True,'diff'] = 0
    df_analyze_results.dropna(inplace=True)

    print("Mistake: %",
          round(100 * len(df_analyze_results[df_analyze_results['direction'] == -1]) / len(df_analyze_results), 2))
    print("Mistake more than +-1%: ", round(100 * len(df_analyze_results[(df_analyze_results['direction'] == -1) & (
            (df_analyze_results['diff'] > 0.01) | (df_analyze_results['diff'] < -0.01))]) / len(df_analyze_results),
                                            2))
    if save_results:
        save_csv(df_analyze_results, param.result_folder + "/"
                                                           f'Analysis for chunk {param.chunk_size} - MA {param.ma_len} - Forecast {param.forecast_size}.csv')

# File Operation

def load_file(file_name: str):
    file_path = param.data_folder + "/" + file_name
    data = pd.read_csv(file_path, index_col=[0], parse_dates=True, infer_datetime_format=True,
                       usecols=param.usable_data_col).rename(
        columns={'RSI.1': 'RSI_OBV', 'True Strength Index': 'TSI', 'Volume MA': 'VOLUME_MA'}).dropna().rename_axis(
        'date').rename(columns=str.upper)
    data['CLOSE_MA'] = data['CLOSE'].rolling(param.ma_len).mean()
    data.dropna(inplace=True)
    return data
def save_csv(data, file_name):
    data.to_csv(file_name)
