import parameters as param
import pandas as pd
class load_files:
    def load_file(file_name: str):
        data = pd.read_csv(file_name, index_col=[0], parse_dates=True, infer_datetime_format=True,
                           usecols=param.usable_data_col).rename(
            columns={'RSI.1': 'RSI_OBV', 'True Strength Index': 'TSI', 'Volume MA': 'VOLUME_MA'}).dropna().rename_axis(
            'date').rename(columns=str.upper)
        data['CLOSE_MA'] = data['CLOSE'].rolling(param.ma_len).mean()
        data.dropna(inplace=True)
        return data