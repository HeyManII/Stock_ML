import os
import pandas as pd
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
    filePath = "./source_data/ML" + stockNumber + ".csv"
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
    data.drop(["index", "High", "Low", "Open", "Adj Close"], axis=1, inplace=True)
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
    id, X, y = [], [], []
    for i in range(0, data.shape[0] - window_size):
        id.append(data.loc[i + window_size, "Date"])
        X.append(data.loc[i : i + window_size - 1, input_cols])
        y.append([data.loc[i + window_size, output_col]])
    return np.array(id), np.array(X), np.array(y)


def build_timeseries_model(
    num_units, window_size, num_features, lr, num_dense_units=16
):
    model = keras.models.Sequential()
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

    # add the specified number of layers
    for i in range(1, len(num_units)):
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=num_units[i],
                    activation="relu",
                    return_sequences=(True if i != len(num_units) - 1 else False),
                )
            )
        )
        # add dropout for regularisation
        model.add(keras.layers.Dropout(0.2))

    # this is a fully-connected layer, where every node from the previous layer is connected to every node in the next layer
    model.add(keras.layers.Dense(units=num_dense_units))
    model.add(keras.layers.Dense(units=1))  # 1 output value

    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        metrics=["mape"],
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
    )
    return model


# helper functions
def get_predicted_prices(
    model,
    features,
    real_prices,
    id_test,
    X_test,
    y_test,
    output_col,
    dataset_cols,
    percent=True,
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
    # predicted_next_prices = y_pred
    real_next_prices = real_prices * (1 + y_test / 100) if percent else y_test
    # real_next_prices = y_test
    decision = pd.DataFrame({"id_test": id_test, "y_pred": y_pred.reshape(-1)})
    # print(
    #     "y_pred:",
    #     y_pred.reshape(
    #         -1,
    #     ),
    # )

    return predicted_next_prices, real_next_prices, decision


# helper functions
def get_predicted_prices_test(
    model,
    features,
    real_prices,
    id_test,
    X_test,
    y_test,
    output_col,
    dataset_cols,
    percent=True,
):
    y_pred = model.predict(X_test).reshape(
        -1,
    )
    decision = pd.DataFrame({"Date": id_test, "y_pred": y_pred.reshape(-1)})

    return predicted_next_prices, decision


def plot_training_history(history):
    history = history.history
    plt.plot(history["val_loss"], label="validation loss")
    plt.plot(history["loss"], label="training loss")
    plt.legend()
    plt.figure()

    plt.plot(history["val_mape"], label="validation MAPE")
    plt.plot(history["mape"], label="training MAPE")
    plt.legend()
    plt.figure()

    if "lr" in history:
        plt.plot(history["lr"], label="learning rate")
        plt.legend()
        plt.figure()


if __name__ == "__main__":
    # Get the stock data
    stock0 = getStockData("2800", "2010-01-01", "2023-10-30")
    stock1 = getStockData("0016", "2010-01-01", "2023-10-30")
    stock2 = getStockData("0002", "2010-01-01", "2023-10-30")
    stock3 = getStockData("0700", "2010-01-01", "2023-10-30")
    stock4 = getStockData("0005", "2010-01-01", "2023-10-30")

    train_stock0 = stock0[stock0["Date"] < "2022-11-01"]
    train_stock1 = stock1[stock1["Date"] < "2022-11-01"]
    train_stock2 = stock2[stock1["Date"] < "2022-11-01"]
    train_stock3 = stock3[stock1["Date"] < "2022-11-01"]
    train_stock4 = stock4[stock1["Date"] < "2022-11-01"]

    test_stock0 = stock0[stock0["Date"] >= "2022-11-01"]
    test_stock1 = stock1[stock1["Date"] >= "2022-11-01"]
    test_stock2 = stock2[stock1["Date"] >= "2022-11-01"]
    test_stock3 = stock3[stock1["Date"] >= "2022-11-01"]
    test_stock4 = stock4[stock1["Date"] >= "2022-11-01"]

    stockname = "0005"

    if stockname == "2800":
        train_stock = train_stock0
        test_stock = test_stock0
    elif stockname == "0016":
        train_stock = train_stock1
        test_stock = test_stock1
    elif stockname == "0002":
        train_stock = train_stock2
        test_stock = test_stock2
    elif stockname == "0700":
        train_stock = train_stock3
        test_stock = test_stock3
    elif stockname == "0005":
        train_stock = train_stock4
        test_stock = test_stock4

    training_stock = dataCleaning(train_stock)

    # add SMA features
    training_stock["Ratio to MA10"] = (
        training_stock["Close"] / training_stock["Close"].rolling(10).mean()
    )
    training_stock["Ratio to MA30"] = (
        training_stock["Close"] / training_stock["Close"].rolling(30).mean()
    )

    # # add SMA features
    training_stock["Ratio to V10"] = (
        training_stock["Volume"] / training_stock["Volume"].rolling(10).mean()
    )
    training_stock["Ratio to V30"] = (
        training_stock["Volume"] / training_stock["Volume"].rolling(30).mean()
    )

    # add % features
    num_days = 10
    for i in range(1, num_days + 1):
        training_stock[f"Day n - {i} Price Change %"] = (
            (training_stock["Close"].shift(i) - training_stock["Close"])
            * 100
            / training_stock["Close"]
        )

    # add a target column
    training_stock["Next 5 days Price"] = training_stock["Close"].shift(-5)
    training_stock["Next 5 days Price Change %"] = (
        (training_stock["Next 5 days Price"] - training_stock["Close"])
        * 100
        / training_stock["Close"]
    )
    training_stock = training_stock.dropna().reset_index(drop=True)

    print(training_stock.info())

    # define variables
    features = [
        # "Close",
        "Day n - 1 Price Change %",
        "Day n - 2 Price Change %",
        "Day n - 3 Price Change %",
        "Day n - 4 Price Change %",
        "Day n - 5 Price Change %",
        "Day n - 6 Price Change %",
        "Day n - 7 Price Change %",
        "Day n - 8 Price Change %",
        "Day n - 9 Price Change %",
        "Day n - 10 Price Change %",
        "Ratio to MA10",
        "Ratio to MA30",
        "Ratio to V10",
        "Ratio to V30",
    ]  # [c for c in data.columns if ("Day" in c or "Ratio" in c)] + ["Close"]
    train_ratio = 0.8
    window_size = 10
    batch_size = 64
    epochs = 150
    normalise = False
    percent = True
    training_set_size = int(training_stock.shape[0] * train_ratio)
    output_col = "Next 5 days Price Change %"
    dataset_cols = list(set(["Date"] + features + [output_col]))
    real_prices = training_stock.loc[training_set_size + window_size :, "Close"]

    # prepare the training dataset
    training_dataset = training_stock[:training_set_size][dataset_cols].reset_index(
        drop=True
    )
    test_dataset = training_stock[training_set_size:][dataset_cols].reset_index(
        drop=True
    )

    # rearrange input data into windows
    id_train, X_train, y_train = split_into_windows(
        training_dataset, window_size, output_col, features
    )

    # repeat the same thing on the test set
    id_test, X_test, y_test = split_into_windows(
        test_dataset, window_size, output_col, features
    )

    # print training set shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # define model structure
    model_type = "bilstm"
    model = build_timeseries_model(
        num_units=[50],
        window_size=window_size,
        num_features=len(features),
        lr=1e-4,
    )

    # start training
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    plot_training_history(history)

    # run the model
    predicted_next_prices, real_next_prices, decision = get_predicted_prices(
        model,
        features,
        real_prices,
        id_test,
        X_test,
        y_test,
        output_col,
        dataset_cols,
        percent=percent,
    )

    # calculate total error
    print(
        "Error of " + model_type.upper() + " regression model:",
        mean_absolute_error(predicted_next_prices, real_next_prices),
    )

    # plot predicted prices
    plot_prices(predicted_next_prices, real_next_prices, model_type.upper())
    print(decision)

    # -------------------- testing --------------------
    testing_stock = dataCleaning(test_stock)

    # add SMA features
    testing_stock["Ratio to MA10"] = (
        testing_stock["Close"] / testing_stock["Close"].rolling(10).mean()
    )
    testing_stock["Ratio to MA30"] = (
        testing_stock["Close"] / testing_stock["Close"].rolling(30).mean()
    )

    # # add SMA features
    testing_stock["Ratio to V10"] = (
        testing_stock["Volume"] / testing_stock["Volume"].rolling(10).mean()
    )
    testing_stock["Ratio to V30"] = (
        testing_stock["Volume"] / testing_stock["Volume"].rolling(30).mean()
    )

    # add % features
    num_days = 10
    for i in range(1, num_days + 1):
        testing_stock[f"Day n - {i} Price Change %"] = (
            (testing_stock["Close"].shift(i) - testing_stock["Close"])
            * 100
            / testing_stock["Close"]
        )

    # add a target column
    testing_stock["Next 5 days Price"] = testing_stock["Close"].shift(-5)
    testing_stock["Next 5 days Price Change %"] = (
        (testing_stock["Next 5 days Price"] - testing_stock["Close"])
        * 100
        / testing_stock["Close"]
    )
    testing_stock = testing_stock.dropna().reset_index(drop=True)

    # define variables
    test_ratio = 1
    testing_set_size = int(testing_stock.shape[0] * test_ratio)
    real_prices1 = testing_stock.loc[testing_set_size + window_size :, "Close"]

    testing_dataset = testing_stock[:testing_set_size][dataset_cols].reset_index(
        drop=True
    )

    # repeat the same thing on the test set
    id_test1, X_test1, y_test1 = split_into_windows(
        testing_dataset, window_size, output_col, features
    )

    print("X_test shape:", X_test1.shape)
    print("y_test shape:", y_test1.shape)

    # run the model
    predicted_next_prices, decision1 = get_predicted_prices_test(
        model,
        features,
        real_prices1,
        id_test1,
        X_test1,
        y_test1,
        output_col,
        dataset_cols,
        percent=percent,
    )

    decision1["Action"] = [
        "SELL" if pred < 0 else "BUY" for pred in decision1["y_pred"]
    ]
    decision1.drop(["y_pred"], axis=1, inplace=True)
    # print(decision1)

    decision1 = decision1[decision1.index % 5 == 0]
    decision1.to_csv(f"{stockname}_MLtrade_decision.csv", index=False)
