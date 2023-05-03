import transformer
import util
import plots
import optuna
import parameters as param
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ORIGINAL_TRANSFORMER_SETTING = {"epoc": 1,
#                                 "num_heads": 1,#4,
#                                 "head_size": 256,
#                                 "ff_dim": 4,
#                                 "num_transformer_blocks": 4,
#                                 "mlp_units": 128,
#                                 "dropout": 0.4,
#                                 "mlp_dropout": 0.25,
#                                 "optimizer_choice": 'adam',
#                                 "loss": 'mean_squared_error',
#                                 "metrics": 'mean_absolute_error',
#                                 "learning_rate": 0.001,
#                                 "min_learning_rate": 0.00001,
#                                 "print_summary": True,
#                                 "validation_split": 0.2,
#                                 "batch_size": 32}
# TRANSFORMER_SETTING = {'epoc': 4, 'optimizer_choice': 'adam', 'num_heads': 4, 'head_size': 256, 'ff_dim': 3,
#                        'num_transformer_blocks': 3, 'mlp_units': 512, 'dropout': 0.2, 'mlp_dropout': 0.5,
#                        'learning_rate': 0.00092, 'validation_split': 0.5, 'batch_size': 32}

# Optimized using Optuna 3 May 2023
TRANSFORMER_SETTING = {'epoc': 3, 'optimizer_choice': 'nadam', 'num_heads': 5, 'head_size': 512, 'ff_dim': 5,
                       'num_transformer_blocks': 2, 'mlp_units': 256, 'dropout': 0.5, 'mlp_dropout': 0.1,
                       'learning_rate': 0.00834, 'validation_split': 0.4,
                       'batch_size': 64}  # Best is trial 47 with value: 0.005554900970309973}


# [I 2023-05-03 04:11:42,190] Trial 47 finished with value: 0.005554900970309973 and parameters: {'optimizer_choice': 'nadam', 'num_heads': 5, 'head_size': 512, 'ff_dim': 5, 'num_transformer_blocks': 2, 'mlp_units': 256, 'dropout': 0.5, 'mlp_dropout': 0.1, 'learning_rate': 0.00834, 'validation_split': 0.4, 'batch_size': 64}. Best is trial 47 with value: 0.005554900970309973.

training_cut_off_date = pd.to_datetime('2019-01-03 09:30:00-05:00')
#
X, Y, X_TEST, Y_TEST, train_mean, train_std, test_mean, test_std, x_0_train, x_0_test = util.gen_multiple_sliding_window(
        param.files, param.chunk_size,
        param.z_normalize,
        training_cut_off_date, 'CLOSE')

def objective(trial):
    idx = np.random.permutation(len(X))
    X_train = X[idx]
    y_train = Y[idx]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_TEST.reshape(X_TEST.shape[0], X_TEST.shape[1], 1)
    # data = util.load_file('data/BATS_SPY.csv')
    # X, y = util.gen_sliding_window(data, param.chunk_size, param.z_normalize)
    # X_train, X_test, y_train, y_test = util.generate_random_sets(util.load_file('data/BATS_SPY.csv'), len_test=300,
    #                                                              test_pct=0.3)

    optimizer = trial.suggest_categorical("optimizer_choice",
                                          ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl'])
    num_head = trial.suggest_int("num_heads", 1, 5)
    head_size = trial.suggest_categorical("head_size", [128, 256, 512])
    ff_dim = trial.suggest_int("ff_dim", 1, 5)
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 1, 5)
    mlp_units = trial.suggest_categorical("mlp_units", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.1, 0.6, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, step=0.00001)
    validation_split = trial.suggest_float("validation_split", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    TRANSFORMER_SETTING["optimizer_choice"] = optimizer
    TRANSFORMER_SETTING["num_heads"] = num_head
    TRANSFORMER_SETTING["head_size"] = head_size
    TRANSFORMER_SETTING["ff_dim"] = ff_dim
    TRANSFORMER_SETTING["num_transformer_blocks"] = num_transformer_blocks
    TRANSFORMER_SETTING["mlp_units"] = mlp_units
    TRANSFORMER_SETTING["dropout"] = dropout
    TRANSFORMER_SETTING["mlp_dropout"] = mlp_dropout
    TRANSFORMER_SETTING["learning_rate"] = learning_rate
    TRANSFORMER_SETTING["validation_split"] = validation_split
    TRANSFORMER_SETTING["print_summary"] = False
    TRANSFORMER_SETTING["batch_size"] = batch_size

    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, **TRANSFORMER_SETTING)
    return transformer.evaluate_model(model, X_test, Y_TEST)


def optimizer():
    study = optuna.create_study(study_name="Transformer Optimization", direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f"Best params: {best_params}")


def main():
    training_cut_off_date = pd.to_datetime('2019-01-03 09:30:00-05:00')

    X_train, y_train, X_test, y_test, train_mean, train_std, test_mean, test_std, x_0_train, x_0_test = \
        util.gen_multiple_sliding_window(
            param.files, param.chunk_size,
            param.z_normalize,
            training_cut_off_date, 'CLOSE')

    # Shuffle data
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    plots.plot_hist_y_distribution(y_train, y_test)

    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, **TRANSFORMER_SETTING)
    transformer.evaluate_model(model, X_test, y_test)

    y_pred_normal = model.predict(X_test)

    if param.z_normalize:
        x_0_test = None
        # y_pred = util.convert_to_original(y_pred_normal, test_mean, test_std)
        # y_test = util.convert_to_original(y_test, test_mean, test_std)
        # X_test = util.convert_to_original(X_test, test_mean, test_std)
    # else:
    y_pred = util.convert_to_original(y_pred_normal, test_mean, test_mean, x_0_test)
    y_test = util.convert_to_original(y_test, test_mean, test_std, x_0_test)
    X_test = util.convert_to_original(X_test, test_mean, test_std, x_0_test)

    plots.plot_scatter_true_vs_predicted(y_test, y_pred, 0, len(y_test)//3)
    plots.plot_histogram_y_test_minus_y_pred(y_test, y_pred, X_test)
    plots.plot_scatter_true_vs_predicted_diagonal(y_test, y_pred)
    plots.plot_scatter_true_vs_predicted_diagonal_only_different_sign(y_test, y_pred)
    util.analyze_results(y_test, y_pred, X_test)


# Call the main function
if __name__ == "__main__":
    # main()
    optimizer()
#
