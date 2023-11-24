import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

SHORT_SMA_DAY = 20
LONG_SMA = 100
RSI_DAY = 14
RSI_SELL_INDEX = 70
RSI_BUY_INDEX = 30
INITIAL_AMOUNT_STOCK = 1


def getStockData(stockNumber, startTime, endTime):
    print("stock number ", stockNumber)
    filePath = "./" + stockNumber + ".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        stockData = yf.download(stockNumber+".HK", start=startTime, end=endTime)
        stockData.reset_index(inplace=True)
        stockData.to_csv(filePath, index=False)
    return stockData

def generateParticularStockDataWithDiagram(stockNumber, startTime, endTime):
    stockData = getStockData(stockNumber, startTime, endTime)


def main():
    startTime = "2015-01-01"
    endTime = "2023-05-31"
    generateParticularStockDataWithDiagram("2800", startTime, endTime)
    generateParticularStockDataWithDiagram("0700", startTime, endTime)

if __name__ == "__main__":
    main()
