chunk_size = 30
forecast_size = 1
ma_len = 1
# Moving Average Length
price_col_name = 'CLOSE'

usable_data_col = ['time', 'close', 'open', 'Volume', 'Volume MA', 'RSI', 'CCI',
       'MACD', 'RSI.1', 'True Strength Index']

files = ['BATS_SPY.csv', 'BATS_QQQ.csv']

# files = ['BATS_XLE.csv' , 'BATS_XLF.csv', 'BATS_XLI.csv', 'BATS_XLK.csv', 'BATS_XLP.csv',
#                'BATS_XLU.csv', 'BATS_XLV.csv', 'BATS_XLY.csv', 'BATS_DIA.csv', 'BATS_IWM.csv', 'BATS_QQQ.csv', 'BATS_SPY.csv']
# 'BATS_XLB.csv', 'BATS_XLRE.csv', 'BATS_XLC.csv'
# files = ['BATS_XLC.csv']

data_folder = "data"
result_folder = "results"
models_folder = "models"
z_normalize = True

plot_file_details = f"chunk {chunk_size} - MA {ma_len} - Forecast {forecast_size}.png"