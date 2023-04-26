chunk_size = 30
forecast_size = 20
ma_len = 5
# Moving Average Length
price_col_name = 'CLOSE'

usable_data_col = ['time', 'close', 'Volume', 'Volume MA', 'RSI', 'CCI',
       'MACD','RSI.1', 'True Strength Index']

files = ['BATS_SPY.csv', 'BATS_QQQ.csv']

# files = ['BATS_XLB.csv', 'BATS_XLE.csv' , 'BATS_XLF.csv', 'BATS_XLI.csv', 'BATS_XLK.csv', 'BATS_XLP.csv', 'BATS_XLRE.csv',
              #  'BATS_XLU.csv', 'BATS_XLV.csv', 'BATS_XLY.csv', 'BATS_DIA.csv', 'BATS_IWM.csv', 'BATS_QQQ.csv', 'BATS_SPY.csv']

result_folder = "results"
models_folder = "models"

plot_file_details = f"chunk {chunk_size} - MA {ma_len} - Forecast {forecast_size}.png"