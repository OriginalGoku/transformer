import pandas as pd
import numpy as np
import parameters as param
class PreProcessor:
    def gen_sliding_window(data: pd.DataFrame, col_X: str, col_y: str):
        # data
        X = np.array([(data[col_X][i:i + param.chunk_size].values / data[col_X][i]) - 1 for i in
                      range(len(data) - param.chunk_size - param.forecast_size)])

        # Create the target array (y_target) containing the first element after the end of the sliding window
        y = np.array(
            [(np.mean(data[col_y][i + param.chunk_size:i + param.chunk_size + param.forecast_size]) / data[col_y][i]) - 1 for i in
             range(len(data) - param.chunk_size - param.forecast_size)])
        return X, y