import transformer
import util
import plots

TRANSFORMER_SETTING = {"epoc":2,
                        "head_size":256,
                          "num_heads":4,
                          "ff_dim":4,
                          "num_transformer_blocks":4,
                          "mlp_units":128,
                          "dropout":0.4,
                          "mlp_dropout":0.25,
                          "optimizer_choice":'adam',
                          "loss":'mean_squared_error',
                          "metrics":'mean_absolute_error',
                          "learning_rate":0.001,
                          "min_learning_rate":0.00001,
                          "print_summary":True,
                          "validation_split":0.2,
                          "batch_size":32}

def main():
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
    util.analyze_results(y_test, y_pred)


# Call the main function
if __name__ == "__main__":
    main()
