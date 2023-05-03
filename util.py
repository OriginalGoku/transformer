import pandas as pd
import numpy as np
import random
import parameters as param
import matplotlib.pyplot as plt
import keras


# Original Sliding Window
# def gen_sliding_window(data: pd.DataFrame, col_X: str, col_y: str,
#                        z_normalize_X: bool, z_normalize_y: bool,
#                        return_mean_y: bool):
#     X, y = [], []
#     n = len(data) - chunk_size - forecast_size
#
#     for i in range(n):
#         chunk_X = data[col_X][i:i + chunk_size]
#         chunk_y = data[col_y][i + chunk_size:i + chunk_size + forecast_size]
#
#         if z_normalize_X:
#             mean_X, std_X = chunk_X.mean(), chunk_X.std()
#             X.append((chunk_X.values - mean_X) / std_X)
#         else:
#             X.append((chunk_X.values / data[col_X][i]) - 1)
#
#         if return_mean_y:
#             mean_y = chunk_y.mean()
#         else:
#             mean_y = chunk_y.iloc[-1]
#
#         if z_normalize_y:
#             y.append((mean_y - mean_X) / std_X)
#         else:
#             y.append((mean_y / data[col_y][i]) - 1)
#
#     return X, y


def gen_sliding_window(data, window_size, z_normalize):
    print(f"Generating Sliding Window (window_size = {window_size}, z_normalize = {z_normalize})")
    sliding_window = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size,))
    n_windows = len(sliding_window)

    sliding, sliding_y, mean_data, std_data = [], [], [], []

    for i in range(n_windows):
        window = sliding_window[i]
        if z_normalize:
            mean = np.mean(window[:-1])
            std = np.std(window[:-1])

            normalized_window = np.round((window - mean) / std, 4)

            mean_data.append(mean)
            std_data.append(std)
        else:
            normalized_window = np.round(100 * ((window / window[0]) - 1), 4)

        sliding.append(normalized_window[:-1])
        sliding_y.append(normalized_window[-1])

    return sliding, sliding_y, mean_data, std_data

def convert_normalized_data(X, y, mean_data, std_data):
    original_X = [X[i] * std_data[i] + mean_data[i] for i in range(len(X))]
    original_y = [y[i] * std_data[i] + mean_data[i] for i in range(len(y))]
    return np.array(original_X), np.array(original_y)


def load_file(file_name: str):
    file_path = param.data_folder + "/" + file_name
    data = pd.read_csv(file_path, index_col=[0], parse_dates=True, infer_datetime_format=True,
                       usecols=param.usable_data_col).rename(
        columns={'RSI.1': 'RSI_OBV', 'True Strength Index': 'TSI', 'Volume MA': 'VOLUME_MA'}).dropna().rename_axis(
        'date').rename(columns=str.upper)
    data['CLOSE_MA'] = data['CLOSE'].rolling(param.ma_len).mean()
    data.dropna(inplace=True)
    return data

def gen_multiple_sliding_window(file_list, window_size, z_normalize, train_cut_off_date, col_name):
    X_train, y_train, X_test, y_test, train_mean, train_std, test_mean, test_std, symbol_file = [], [], [], [], [], [], [], [], []
    for symbol_file in file_list:
        print(f"Processing {symbol_file}")
        data = load_file(symbol_file)
        data_train = data[data.index < train_cut_off_date]
        data_test = data[data.index >= train_cut_off_date]
        X_train_temp, y_train_temp, train_mean, train_std = gen_sliding_window(data_train[col_name], window_size, z_normalize)
        X_test_temp, y_test_temp, test_mean, test_std = gen_sliding_window(data_test[col_name], window_size, z_normalize)
        X_train.extend(X_train_temp)
        y_train.extend(y_train_temp)
        X_test.extend(X_test_temp)
        y_test.extend(y_test_temp)
        train_mean.extend(train_mean)
        train_std.extend(train_std)
        test_mean.extend(test_mean)
        test_std.extend(test_std)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(train_mean), np.array(train_std), np.array(test_mean), np.array(test_std)

def load_model(model_name: str):
    model = keras.models.load_model(model_name)
    return model


def analyze_results(y_test, y_pred, save_results=True):
    df_analyze_results = pd.DataFrame()
    df_analyze_results['original'] = y_test.flatten()
    df_analyze_results['predict'] = y_pred
    df_analyze_results['direction'] = np.sign(
        df_analyze_results['original'] * df_analyze_results['predict'])  # .astype(bool)
    df_analyze_results['diff'] = df_analyze_results['original'] - df_analyze_results['predict']
    df_analyze_results.dropna(inplace=True)

    print("Mistake: %",
          round(100 * len(df_analyze_results[df_analyze_results['direction'] == -1]) / len(df_analyze_results), 2))
    print("Mistake more than +-1%: ", round(100 * len(df_analyze_results[(df_analyze_results['direction'] == -1) & (
            (df_analyze_results['diff'] > 0.01) | (df_analyze_results['diff'] < -0.01))]) / len(df_analyze_results),
                                            2))
    if save_results:
        save_csv(df_analyze_results, param.result_folder + "/"
                                                           f'Analysis for chunk {param.chunk_size} - MA {param.ma_len} - Forecast {param.forecast_size}.csv')


def save_csv(data, file_name):
    data.to_csv(file_name)
