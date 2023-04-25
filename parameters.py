chunk_size = 30
forecast_size = 20
ma_len = 5
# Moving Average Length
price_col_name = 'CLOSE'

usable_data_col = ['time', 'close', 'Volume', 'Volume MA', 'RSI', 'CCI',
       'MACD','RSI.1', 'True Strength Index']

files = ['BATS_SPY.csv', 'BATS_QQQ.csv']