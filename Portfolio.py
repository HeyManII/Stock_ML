import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

def getStockData(stockNumber, startTime, endTime):
    print("stock number ", stockNumber)
    filePath = "./" + stockNumber + ".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        print("downloading data")
        stockData = yf.download(stockNumber + ".HK", start=startTime, end=endTime)
        stockData.to_csv(filePath, index=False)
    return stockData

if __name__ == "__main__":
    