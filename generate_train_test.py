import load_files as file_loader
import parameters as param
import pre_processing as PreProcessor
import numpy as np
class StockDataGenerator:
    def load_all_data(self, files):
        # X, y = [], []

        X = np.array((1,))

        for file_name in files:
            print(f"Loading {file_name}")
            data = file_loader.load_file(file_name)
            print(f"Loaded {len(data)} rows of data")
            X_temp, y_temp = PreProcessor.gen_sliding_window(data, 'CLOSE_MA', 'CLOSE')
            X = np.concatenate((X, X_temp))
            y = np.concatenate((y, y_temp))

    def generate_train_test(self): # X = X_rsi.reshape(X_rsi.shape[0],X_rsi.shape[1],1)
        X = X_price.reshape(X_price.shape[0],X_price.shape[1],1)


        # assuming your data is stored in X (features) and y (target variable)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)