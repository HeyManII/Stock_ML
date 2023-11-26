import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
from constant import DEFAULT_STOCK_LIST

def getStockData(stockNumber, startTime, endTime, interval):
    print("stock number ", stockNumber)
    filePath = "./source_data/" + stockNumber + ".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        stockData = yf.download(stockNumber+".HK", start=startTime, end=endTime, interval=interval)
        stockData.reset_index(inplace=True)
        stockData.to_csv(filePath, index=False)
    return stockData

def generateParticularStockDataWithDiagram(stockNumber, startTime, endTime):
    getStockData(stockNumber, startTime, endTime, '1wk')


def main():
    startTime = "2015-01-01"
    endTime = "2023-10-31"
    for stockNumber in DEFAULT_STOCK_LIST:
        generateParticularStockDataWithDiagram(stockNumber, startTime, endTime)

if __name__ == "__main__":
    main()
