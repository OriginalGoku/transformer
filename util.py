import pandas as pd
import numpy as np
import random
import parameters as param
import matplotlib.pyplot as plt
import keras


def gen_sliding_window(data: pd.DataFrame(), col_X: str, col_y: str, z_normalize: bool = False):
    # data
    if z_normalize:
        X = [((data[col_X][i:i + param.chunk_size].values) - np.mean(data[col_X][i:i + param.chunk_size].values)) / (
            np.std(data[col_X][i:i + param.chunk_size].values)) for i in
             range(len(data) - param.chunk_size - param.forecast_size)]
    else:
        X = np.array([(data[col_X][i:i + param.chunk_size].values / data[col_X][i]) - 1 for i in
                      range(len(data) - param.chunk_size - param.forecast_size)])

    # Create the target array (y_target) containing the first element after the end of the sliding window
    y = np.array(
        [(np.mean(data[col_y][i + param.chunk_size:i + param.chunk_size + param.forecast_size]) / data[col_y][i]) - 1
         for i in
         range(len(data) - param.chunk_size - param.forecast_size)])

    return X, y


def generate_random_sets_CHAT_GPT_NOT_WORKING(data: pd.DataFrame(), len_test: int = 300, test_pct: float = 0.3,
                                              z_normalize: bool = False, x_col='CLOSE_MA', y_col='CLOSE',
                                              return_3d_data=True):
    print("Generating random sets...")
    X_random_train = []
    y_random_train = []
    X_random_test = []
    y_random_test = []

    current_pointer = 0
    chunk_len = int(len_test / (1 - test_pct))
    train_chunk_len = chunk_len - len_test

    for counter in range((len(data) - param.chunk_size - param.forecast_size) // chunk_len):
        next_point = random.randint(0, train_chunk_len - 1 - param.chunk_size - param.forecast_size) + (
                counter * chunk_len)
        print(f"next_point: {next_point}")
        print(f"current_pointer: {current_pointer}")

        x_train_temp, y_train_temp = gen_sliding_window(data.iloc[current_pointer: next_point], x_col, y_col,
                                                        z_normalize)
        X_random_train.extend(x_train_temp)
        y_random_train.extend(y_train_temp)
        x_test_temp, y_test_temp = gen_sliding_window(data.iloc[next_point:next_point + len_test], x_col,
                                                      y_col, z_normalize)
        print(f"len(x_test_temp): {len(x_test_temp)}")
        X_random_test.extend(x_test_temp)
        y_random_test.extend(y_test_temp)
        current_pointer = next_point + len_test
        print("------")

    X_random_train = np.array(X_random_train)
    y_random_train = np.array(y_random_train)
    X_random_test = np.array(X_random_test)
    y_random_test = np.array(y_random_test)

    if return_3d_data:
        return X_random_train.reshape(*X_random_train.shape, 1), y_random_train, \
            X_random_test.reshape(*X_random_test.shape, 1), y_random_test
    else:
        return X_random_train, y_random_train, X_random_test, y_random_test


def generate_random_sets(data: pd.DataFrame(), len_test: int = 300,
                                                         test_pct: float = 0.3,
                                                         z_normalize: bool = False, x_col='CLOSE_MA', y_col='CLOSE',
                                                         return_3d_data=True):
    print("Generating random sets...")
    X_random_train = []
    y_random_train = []
    X_random_test = []
    y_random_test = []

    current_pointer = 0
    chunk_len = int(len_test / test_pct)
    for counter in range(len(data) // chunk_len):
        next_point = random.randint(0, chunk_len - len_test) + (counter * chunk_len)
        # print(f"next_point: {next_point}")
        # print(f"current_pointer: {current_pointer}")
        # print(f"len_test: {len_test}")
        # todo: keep the date ranges for train and test for future analysis
        x_train_temp, y_train_temp = gen_sliding_window(data.iloc[current_pointer: next_point], x_col, y_col,
                                                        z_normalize)
        X_random_train.extend(x_train_temp)
        y_random_train.extend(y_train_temp)
        x_test_temp, y_test_temp = gen_sliding_window(data.iloc[next_point:next_point + len_test], x_col,
                                                      y_col, z_normalize)
        # print(f"len(x_test_temp): {len(x_test_temp)}")
        X_random_test.extend(x_test_temp)
        y_random_test.extend(y_test_temp)
        current_pointer = next_point + len_test

    X_random_train = np.array(X_random_train)
    y_random_train = np.array(y_random_train)
    X_random_test = np.array(X_random_test)
    y_random_test = np.array(y_random_test)
    print(f"Generated {len(X_random_train)} Training and {len(X_random_test)} Test Data points")
    if return_3d_data:
        return X_random_train.reshape(*X_random_train.shape, 1), X_random_test.reshape(*X_random_test.shape, 1), \
            y_random_train, y_random_test
    else:
        return X_random_train, X_random_test, y_random_train, y_random_test


def generate_random_sets1(data: pd.DataFrame(), len_test: int = 300, test_pct: float = 0.3, z_normalize: bool = False,
                         x_col='CLOSE_MA', y_col='CLOSE'):
    X_random_train = []
    y_random_train = []
    X_random_test = []
    y_random_test = []

    current_pointer = 0
    len_test = 300
    for _ in range(len(data) // 1000):
        next_point = random.randint(0, 699) + _ * 1000
        print(next_point)
        x_train_temp, y_train_temp = gen_sliding_window(data.iloc[current_pointer: next_point], 'CLOSE_MA', 'CLOSE',
                                                        z_normalize)
        X_random_train.extend(x_train_temp)
        y_random_train.extend(y_train_temp)
        x_test_temp, y_test_temp = gen_sliding_window(data.iloc[next_point:next_point + len_test], 'CLOSE_MA', 'CLOSE',
                                                      z_normalize)
        X_random_test.extend(x_test_temp)
        y_random_test.extend(y_test_temp)
        # print(f"len(x_test_temp): {len(x_test_temp)}")
        # print(f"len(X_random_test): {len(X_random_test)}")
        current_pointer = next_point + len_test

    X_random_train = np.array(X_random_train)
    y_random_train = np.array(y_random_train)
    X_random_test = np.array(X_random_test)
    # print("--------")
    print(f"len(X_random_test): {len(X_random_test)}")
    y_random_test = np.array(y_random_test)
    # if return_3d_data:
    return X_random_train.reshape(*X_random_train.shape, 1), X_random_test.reshape(*X_random_test.shape, 1), \
        y_random_train, y_random_test
    # else:
    #     return X_random_train, y_random_train, X_random_test, y_random_test


def load_file(file_name: str):
    data = pd.read_csv(file_name, index_col=[0], parse_dates=True, infer_datetime_format=True,
                       usecols=param.usable_data_col).rename(
        columns={'RSI.1': 'RSI_OBV', 'True Strength Index': 'TSI', 'Volume MA': 'VOLUME_MA'}).dropna().rename_axis(
        'date').rename(columns=str.upper)
    data['CLOSE_MA'] = data['CLOSE'].rolling(param.ma_len).mean()
    data.dropna(inplace=True)
    return data


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
