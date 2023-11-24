import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def getStockData(stockNumber, startTime, endTime):
    print("stock number ", stockNumber)
    filePath = "./ML" + stockNumber + ".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        stockData = yf.download(stockNumber + ".HK", start=startTime, end=endTime)
        stockData.reset_index(inplace=True)
        stockData.to_csv(filePath, index=False)
    return stockData


def dataCleaning(data):
    data = data.reset_index().dropna()
    data.reset_index()
    data.drop(["High", "Low", "Open", "Adj Close"], axis=1, inplace=True)
    return data


# prepare a helper function for plotting prices to avoid code redundancy
def plot_prices(y_pred, y_test, model_name, title="Closing Price Predictions"):
    plt.figure(figsize=(9, 6))
    plt.title(title + f" ({model_name})")
    plt.plot(y_pred, label=model_name)
    plt.plot(y_test, label="Actual")
    plt.ylabel("Price")
    plt.xlabel("Day")
    plt.legend()
    plt.show()


# define helper functions for LSTM training
# this function splits the data into sequences using the sliding window approach
def split_into_windows(data, window_size, output_col, input_cols):
    X, y = [], []
    for i in range(0, data.shape[0] - window_size):
        X.append(data.loc[i : i + window_size - 1, input_cols])
        y.append([data.loc[i + window_size, output_col]])
    return np.array(X), np.array(y)


def build_timeseries_model(
    num_units, window_size, num_features, model_type, lr, num_dense_units=16
):
    if model_type not in ["rnn", "bilstm", "lstm"]:
        raise Exception(f"Unsupported model type {model_type}")

    model = keras.models.Sequential()

    # at least 1 layer is required
    if model_type == "rnn":
        model.add(
            keras.layers.SimpleRNN(
                units=num_units[0],
                input_shape=(window_size, num_features),
                activation="relu",
                return_sequences=(True if len(num_units) > 1 else False),
            )
        )
    elif model_type == "bilstm":
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=num_units[0],
                    input_shape=(window_size, num_features),
                    activation="relu",
                    return_sequences=(True if len(num_units) > 1 else False),
                )
            )
        )
    elif model_type == "lstm":
        model.add(
            keras.layers.LSTM(
                units=num_units[0],
                input_shape=(window_size, num_features),
                activation="relu",
                return_sequences=(True if len(num_units) > 1 else False),
            )
        )

    # add the specified number of layers
    for i in range(1, len(num_units)):
        if model_type == "rnn":
            model.add(
                keras.layers.SimpleRNN(
                    units=num_units[i],
                    activation="relu",
                    return_sequences=(True if i != len(num_units) - 1 else False),
                )
            )
        elif model_type == "bilstm":
            model.add(
                keras.layers.Bidirectional(
                    keras.layers.LSTM(
                        units=num_units[i],
                        activation="relu",
                        return_sequences=(True if i != len(num_units) - 1 else False),
                    )
                )
            )
        elif model_type == "lstm":
            model.add(
                keras.layers.LSTM(
                    units=num_units[i],
                    activation="relu",
                    return_sequences=(True if i != len(num_units) - 1 else False),
                )
            )

        # add dropout for regularisation
        model.add(keras.layers.Dropout(0.2))

    # this is a fully-connected layer, where every node from the previous layer is connected to every node in the next layer
    model.add(keras.layers.Dense(units=num_dense_units))
    model.add(keras.layers.Dense(units=1))  # 1 output value

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mse"],
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
    )
    return model


# helper functions
def get_predicted_prices(
    model, features, real_prices, X_test, y_test, output_col, dataset_cols, percent=True
):
    y_pred = model.predict(X_test).reshape(
        -1,
    )
    y_test = y_test.reshape(
        -1,
    )
    assert y_test.shape == y_pred.shape

    # now let's generate closing price predictions based on the predicted changes
    predicted_next_prices = real_prices * (1 + y_pred / 100) if percent else y_pred
    real_next_prices = real_prices * (1 + y_test / 100) if percent else y_test
    print(
        "y_pred:",
        y_pred.reshape(
            -1,
        ),
    )

    return predicted_next_prices, real_next_prices


def plot_training_history(history):
    history = history.history
    plt.plot(history["val_loss"], label="validation loss")
    plt.plot(history["loss"], label="training loss")
    plt.legend()
    plt.figure()

    plt.plot(history["val_mse"], label="validation MSE")
    plt.plot(history["mse"], label="training MSE")
    plt.legend()
    plt.figure()

    if "lr" in history:
        plt.plot(history["lr"], label="learning rate")
        plt.legend()
        plt.figure()


if __name__ == "__main__":
    # Get the stock data
    stock1 = getStockData("0016", "2015-11-01", "2022-10-30")
    stock2 = getStockData("0002", "2015-11-01", "2022-10-30")
    stock3 = getStockData("0700", "2015-11-01", "2022-10-30")
    stock4 = getStockData("0005", "2015-11-01", "2022-10-30")

    training_stock1 = dataCleaning(stock1)
    training_stock2 = dataCleaning(stock2)
    training_stock3 = dataCleaning(stock3)
    training_stock4 = dataCleaning(stock4)

    # add SMA features
    training_stock1["Ratio to MA10"] = (
        training_stock1["Close"] / training_stock1["Close"].rolling(10).mean()
    )
    training_stock1["Ratio to MA30"] = (
        training_stock1["Close"] / training_stock1["Close"].rolling(30).mean()
    )

    # add % features
    num_days = 5
    for i in range(1, num_days + 1):
        training_stock1[f"Day n - {i} Price Change %"] = (
            (training_stock1["Close"].shift(i) - training_stock1["Close"])
            * 100
            / training_stock1["Close"]
        )
        training_stock1[f"Day n - {i} Volume Change %"] = (
            (training_stock1["Volume"].shift(i) - training_stock1["Volume"])
            * 100
            / training_stock1["Volume"]
        )

    # add a target column
    training_stock1["Next 5 days Price"] = training_stock1["Close"].shift(-5)
    training_stock1["Next 5 days Price Change %"] = (
        (training_stock1["Next 5 days Price"] - training_stock1["Close"])
        * 100
        / training_stock1["Close"]
    )
    training_stock1 = training_stock1.dropna().reset_index(drop=True)

    print(training_stock1.head())
    print(training_stock1.info())

    # # define variables
    # features = [
    #     #     # "Close",
    #     #     # "Volume",
    #     "Ratio to MA10",
    #     "Ratio to MA30",
    #     "Day n - 1 Volume Change %",
    #     "Day n - 2 Volume Change %",
    #     "Day n - 3 Volume Change %",
    #     "Day n - 4 Volume Change %",
    #     "Day n - 5 Volume Change %",
    #     "Day n - 1 Price Change %",
    #     "Day n - 2 Price Change %",
    #     "Day n - 3 Price Change %",
    #     "Day n - 4 Price Change %",
    #     "Day n - 5 Price Change %",
    # ]  # [c for c in data.columns if ("Day" in c or "Ratio" in c)] + ["Close"]
    # train_ratio = 0.85
    # window_size = 10
    # batch_size = 64
    # epochs = 100
    # normalise = False
    # percent = "%" in features[0]
    # training_set_size = int(training_stock1.shape[0] * train_ratio)
    # output_col = "Next 5 days Price Change %"
    # dataset_cols = list(set(features + [output_col]))
    # real_prices = training_stock1.loc[training_set_size + window_size :, "Close"]

    # # prepare the training dataset
    # training_dataset = training_stock1[:training_set_size][dataset_cols].reset_index(
    #     drop=True
    # )
    # test_dataset = training_stock1[training_set_size:][dataset_cols].reset_index(
    #     drop=True
    # )

    # # rearrange input data into windows
    # X_train, y_train = split_into_windows(
    #     training_dataset, window_size, output_col, features
    # )

    # # repeat the same thing on the test set
    # X_test, y_test = split_into_windows(test_dataset, window_size, output_col, features)

    # # print training set shapes
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    # # define model structure
    # model_type = "bilstm"
    # model = build_timeseries_model(
    #     num_units=[50],
    #     window_size=window_size,
    #     num_features=len(features),
    #     model_type=model_type,
    #     lr=1e-3,
    # )

    # # start training
    # history = model.fit(
    #     X_train,
    #     y_train,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     validation_data=(X_test, y_test),
    #     verbose=1,
    # )
    # plot_training_history(history)

    # # run the model
    # predicted_next_prices, real_next_prices = get_predicted_prices(
    #     model,
    #     features,
    #     real_prices,
    #     X_test,
    #     y_test,
    #     output_col,
    #     dataset_cols,
    #     percent=percent,
    # )

    # # calculate total error
    # print(
    #     "Error of " + model_type.upper() + " regression model:",
    #     mean_absolute_error(predicted_next_prices, real_next_prices),
    # )

    # # plot predicted prices
    # plot_prices(predicted_next_prices, real_next_prices, model_type.upper())
