from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random
import parameters as param


# model processes a tensor of shape (batch size, sequence length, features), where sequence length is the number of time steps and features is each input timeseries.
#We include residual connections, layer normalization, and dropout. The resulting layer can be stacked multiple times.
# The projection layers are implemented through keras.layers.Conv1D.
#We can stack multiple of those transformer_encoder blocks and we can also proceed to add the final
# Multi-Layer Perceptron classification or regressionhead. Apart from a stack of Dense layers, we need to reduce the
# output tensor of the TransformerEncoder part of our model down to a vector of features for each data point in the
# current batch. A common way to achieve this is to use a pooling layer. For this example, we used a
# GlobalAveragePooling1D layer but more studies would be good to see if this choice can improve results.
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation=None)(x)
    return keras.Model(inputs, outputs)


def construct_transformer(X_train, y_train, epoc, head_size = 256, num_heads = 4, ff_dim = 4, num_transformer_blocks = 4,
                          mlp_units = 128, dropout = 0.4, mlp_dropout = 0.25, optimizer_choice = 'adam',
                          loss = 'mean_squared_error', metrics = 'mean_absolute_error', learning_rate = 0.001, min_learning_rate = 0.00001,
                          print_summary = True, validation_split = 0.2, batch_size = 32):

    input_shape = X_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=[mlp_units],
        mlp_dropout=mlp_dropout,
        dropout=dropout,
    )

    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if print_summary:
        model.summary()

    model_name = param.models_folder+'/best_model chunk ['+str(param.chunk_size)+"] SMA ["+str(param.ma_len)+"] forecast ["+str(param.forecast_size)+"].h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_name, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=min_learning_rate
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epoc,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    return history, model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test dataset
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)

    print("Test Loss:", test_loss)
    print("Test Mean Absolute Error:", test_mae)
    # Predict the values on the test dataset