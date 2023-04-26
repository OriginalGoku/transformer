import transformer
import util
import plots
import optuna

TRANSFORMER_SETTING = {"epoc": 2,
                       "head_size": 256,
                       "num_heads": 4,
                       "ff_dim": 4,
                       "num_transformer_blocks": 4,
                       "mlp_units": 128,
                       "dropout": 0.4,
                       "mlp_dropout": 0.25,
                       "optimizer_choice": 'adam',
                       "loss": 'mean_squared_error',
                       "metrics": 'mean_absolute_error',
                       "learning_rate": 0.001,
                       "min_learning_rate": 0.00001,
                       "print_summary": True,
                       "validation_split": 0.2,
                       "batch_size": 32}

def objective(trial):
    X_train, X_test, y_train, y_test = util.generate_random_sets(util.load_file('data/BATS_SPY.csv'), len_test=300,
                                                                 test_pct=0.3)

    optimizer = trial.suggest_categorical("optimizer_choice", ["sgd", "adam"])
    num_head = trial.suggest_int("num_heads", 1, 5)
    head_size = trial.suggest_int("head_size", 128, 512, log=True)
    ff_dim = trial.suggest_int("ff_dim", 1, 5)
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 1, 5)
    mlp_units = trial.suggest_int("mlp_units", 128, 512, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.1, 0.6, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, step=0.00002)
    validation_split = trial.suggest_float("validation_split", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_int("batch_size", 16, 64, log=True)



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
    return transformer.evaluate_model(model, X_test, y_test)

def optuna_optimize():
    study = optuna.create_study(study_name="Transformer Optimization", direction="minimize")
    study.optimize(objective, n_trials=2)
    best_params = study.best_params
    print(f"Best params: {best_params}")

def main():
    optuna_optimize()
def main11():
    X_train, X_test, y_train, y_test = util.generate_random_sets(util.load_file('data/BATS_SPY.csv'), len_test=300,
                                                                 test_pct=0.3)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    plots.plot_hist_y_distribution(y_train, y_test)

    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, **TRANSFORMER_SETTING)
    transformer.evaluate_model(model, X_test, y_test)

    y_pred = model.predict(X_test)
    plots.plot_scatter_true_vs_predicted(y_test, y_pred, 100, 200)
    plots.plot_histogram_y_test_minus_y_pred(y_test, y_pred)
    plots.plot_scatter_true_vs_predicted_diagonal(y_test, y_pred)
    plots.plot_scatter_true_vs_predicted_diagonal_only_different_sign(y_test, y_pred)
    util.analyze_results(y_test, y_pred)


# Call the main function
if __name__ == "__main__":
    main()
