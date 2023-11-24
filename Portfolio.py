import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math


def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if "M" in x:
        if len(x) > 1:
            return float(x.replace("M", "")) * 1000000
        return 1000000.0
    if "B" in x:
        return float(x.replace("B", "")) * 1000000000
    return 0.0


def getStockData(stockNumber, startTime, endTime):
    print("stock number ", stockNumber)
    filePath = "./p" + stockNumber + ".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        stockData = yf.download(stockNumber + ".HK", start=startTime, end=endTime)
        stockData.reset_index(inplace=True)
        stockData.to_csv(filePath, index=False)
    return stockData


def getLargetMarketCap(sector):
    df = pd.read_csv("HSI_list.csv")
    df = df[df["Sector"] == sector]
    df["Market Cap"] = df["Market Cap"].apply(value_to_float)
    df_sorted = df.sort_values(by=["Market Cap"], ascending=False)
    chosen = df_sorted["Symbol"].iloc[0]
    return chosen


def dataCleaning(data):
    data = data.reset_index().dropna()
    data.drop(["High", "Low", "Open", "Adj Close", "Volume"], axis=1, inplace=True)
    return data


def stockInfoCalculate(data):
    expectedReturn = data["Return"].mean()
    variance = data["Return"].var()
    std = math.sqrt(variance)
    return expectedReturn, variance, std


if __name__ == "__main__":
    stockNumber = 4

    df = pd.read_csv("HSI_list.csv")
    # Get the stock with largest market cap in HSP
    HSP = getLargetMarketCap("HSP")[1:5]
    HSU = getLargetMarketCap("HSU")[1:5]
    HSC = getLargetMarketCap("HSC")[1:5]
    HSF = getLargetMarketCap("HSF")[1:5]
    print("HSP: ", HSP, "-", np.array(df[df["Symbol"] == "0" + HSP + ".HK"]["Name"])[0])
    print("HSU: ", HSU, "-", np.array(df[df["Symbol"] == "0" + HSU + ".HK"]["Name"])[0])
    print("HSC: ", HSC, "-", np.array(df[df["Symbol"] == "0" + HSC + ".HK"]["Name"])[0])
    print("HSF: ", HSF, "-", np.array(df[df["Symbol"] == "0" + HSF + ".HK"]["Name"])[0])

    HSP_share = 500
    HSU_share = 500
    HSC_share = 100
    HSF_share = 400

    # Get the stock data
    HSP_df = getStockData(HSP, "2015-11-01", "2022-10-30")
    HSU_df = getStockData(HSU, "2015-11-01", "2022-10-30")
    HSC_df = getStockData(HSC, "2015-11-01", "2022-10-30")
    HSF_df = getStockData(HSF, "2015-11-01", "2022-10-30")

    # perform data cleaning
    HSP_df = dataCleaning(HSP_df)
    HSU_df = dataCleaning(HSU_df)
    HSC_df = dataCleaning(HSC_df)
    HSF_df = dataCleaning(HSF_df)

    HSP_df["Return"] = (HSP_df["Close"] - HSP_df["Close"].shift(5)) * HSP_share
    HSU_df["Return"] = (HSU_df["Close"] - HSU_df["Close"].shift(5)) * HSU_share
    HSC_df["Return"] = (HSC_df["Close"] - HSC_df["Close"].shift(5)) * HSC_share
    HSF_df["Return"] = (HSF_df["Close"] - HSF_df["Close"].shift(5)) * HSF_share

    HSP_expected_return, HSP_variance, HSP_std = stockInfoCalculate(HSP_df)
    HSU_expected_return, HSU_variance, HSU_std = stockInfoCalculate(HSU_df)
    HSC_expected_return, HSC_variance, HSC_std = stockInfoCalculate(HSC_df)
    HSF_expected_return, HSF_variance, HSF_std = stockInfoCalculate(HSF_df)

    df1 = pd.concat(
        [HSP_df["Close"], HSU_df["Close"], HSC_df["Close"], HSF_df["Close"]],
        axis=1,
    )
    df1.columns = [HSP, HSU, HSC, HSF]
    covariance = df1.cov()

    # initial weights
    possible_values = []
    for w1 in range(10, 71, 5):
        for w2 in range(10, 71, 5):
            for w3 in range(10, 71, 5):
                w4 = 100 - w1 - w2 - w3
                if w4 >= 10:
                    possible_values.append((w1, w2, w3, w4))

    expected_return_values = []
    std_values = []
    df2 = pd.DataFrame(
        [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
        columns=["w1", "w2", "w3", "w4", "expected_return", "std"],
    )
    for w1, w2, w3, w4 in possible_values:
        w1 = w1 / 100
        w2 = w2 / 100
        w3 = w3 / 100
        w4 = w4 / 100
        weight = [w1, w2, w3, w4]
        expected_return = (
            HSP_expected_return * w1
            + HSU_expected_return * w2
            + HSC_expected_return * w3
            + HSF_expected_return * w4
        )
        variance = 0
        for a in range(4):
            for b in range(4):
                variance += weight[a] * weight[b] * covariance.iloc[a, b]
        std = math.sqrt(variance)
        expected_return_values.append(expected_return)
        std_values.append(std)
        df_temp = pd.DataFrame(
            [[w1, w2, w3, w4, expected_return, std]],
            columns=["w1", "w2", "w3", "w4", "expected_return", "std"],
        )
        df2 = pd.concat([df2, df_temp], ignore_index=True)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Plot the graph
    plt.plot(std_values, expected_return_values)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Expected Return")
    plt.title("Portfolio Optimization")
    plt.show()

# Find the row where std is minimum and expected return is maximum
min_std_row = df2[(df2["std"] == df2["std"].min())]
print(min_std_row)
print(
    "w1 - HSP: ", HSP, "-", np.array(df[df["Symbol"] == "0" + HSP + ".HK"]["Name"])[0]
)
print(
    "w2 - HSU: ", HSU, "-", np.array(df[df["Symbol"] == "0" + HSU + ".HK"]["Name"])[0]
)
print(
    "w3 - HSC: ", HSC, "-", np.array(df[df["Symbol"] == "0" + HSC + ".HK"]["Name"])[0]
)
print(
    "w4 - HSF: ", HSF, "-", np.array(df[df["Symbol"] == "0" + HSF + ".HK"]["Name"])[0]
)
