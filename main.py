import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

SHORT_SMA_DAY = 20
LONG_SMA = 100
RSI_DAY=14
RSI_SELL_INDEX=70
RSI_BUY_INDEX=30
def getStockData(stockNumber,startTime,endTime):
    print("stock number ",stockNumber)
    filePath = "./"+stockNumber+".csv"
    stockData = None
    if os.path.isfile(filePath):
        stockData = pd.read_csv(filePath)
    else:
        # Download QQQ data from 2010 to 2022
        print("downloading data")
        stockData = yf.download(stockNumber+".HK", start=startTime, end=endTime)
        stockData.to_csv(filePath, index=False)
    return stockData

def calculateSmaInStockData(originStockData,day,referenceColumnName):
    newStockData =originStockData
    newStockData["SMA"+str(day)] = newStockData[referenceColumnName].rolling(window=day).mean()
    return newStockData

def createRsiInStockData(originStockData,day):
    newStockData=originStockData
    delta = newStockData['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=day).mean()
    avg_loss = loss.rolling(window=day).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    newStockData["RSI"] = rsi
    return newStockData

def generateParticularStockDataWithDiagram(stockNumber,startTime,endTime):
    taItems = ["SMA","RSI"]
    stockData = getStockData(stockNumber,startTime,endTime)
    stockData = calculateSmaInStockData(stockData,SHORT_SMA_DAY,"Close")
    stockData = calculateSmaInStockData(stockData,LONG_SMA,"Close")
    stockData = createRsiInStockData(stockData,RSI_DAY)
    row = len(taItems)
    column = 1
    fig, ax = plt.subplots(row, column, figsize=(10, 10), sharex=True)
    for index, taItem in enumerate(taItems):
        if taItem == "SMA":
            ax[index].plot(stockData["Close"], color="green", label="Close")
            ax[index].plot(stockData["SMA"+str(SHORT_SMA_DAY)], color="blue", label="SMA"+str(SHORT_SMA_DAY))
            ax[index].plot(stockData["SMA"+str(LONG_SMA)], color="orange", label="SMA"+str(LONG_SMA))
            ax[index].set_title(stockNumber+" open price")
            ax[index].legend()
        elif taItem == "RSI":        
            ax[index].plot(stockData["RSI"], color="green", label="RSI")
            ax[index].axhline(y=RSI_BUY_INDEX, color="red", linestyle="--")
            ax[index].axhline(y=RSI_SELL_INDEX, color="red", linestyle="--")
            ax[index].set_title(stockNumber+" RSI")
            ax[index].legend()
    
    plt.show()
def main():
    startTime = "2015-01-01"
    endTime = "2023-05-31"
    generateParticularStockDataWithDiagram("2800",startTime,endTime)
    generateParticularStockDataWithDiagram("0700",startTime,endTime)
main()





# # step1 pick stock number

# # step2 create csv file 

# if os.path.isfile("./2800.csv"):
#     hk_stock_2800 = pd.read_csv("./2800.csv")
# else:
#     # Download QQQ data from 2010 to 2022
#     print("downloading data")
#     hk_stock_2800 = yf.download("2800.HK", start="2015-01-01", end="2023-05-31")
#     hk_stock_2800.to_csv("./2800.csv", index=False)

# # Print the first 5 rows of the data
# # print(hk_stock_2800.head())

# hk_stock_2800["SMA20"] = hk_stock_2800["Close"].rolling(window=20).mean()
# hk_stock_2800["SMA100"] = hk_stock_2800["Close"].rolling(window=100).mean()
# print("hk_stock_2800",hk_stock_2800)
# delta = hk_stock_2800["Close"].diff()
# gain = delta.where(delta > 0, 0)
# loss = -delta.where(delta < 0, 0)
# avg_gain = gain.rolling(window=14).mean()
# avg_loss = loss.rolling(window=14).mean()
# rs = avg_gain / avg_loss
# rsi = 100 - (100 / (1 + rs))

# fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# # ax[0].plot(hk_stock_2800["Open"], color="red", label="Open")
# ax[0].plot(hk_stock_2800["Close"], color="green", label="Close")
# ax[0].plot(hk_stock_2800["SMA20"], color="blue", label="SMA20")
# ax[0].plot(hk_stock_2800["SMA100"], color="orange", label="SMA100")
# ax[0].set_title("2800 open price")
# ax[0].legend()

# ax[1].plot(hk_stock_2800.index, rsi, label="RSI")
# ax[1].axhline(y=30, color="red", linestyle="--")
# ax[1].axhline(y=70, color="red", linestyle="--")
# ax[1].set_title("2800 RSI")
# ax[1].legend()


# # add red circle when SMA20 = SMA100
# sma20 = hk_stock_2800["SMA20"]
# sma100 = hk_stock_2800["SMA100"]
# for i in range(1, len(sma20)):
#     if sma20[i] == sma100[i]:
#         if rsi[i] < 70:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
#         else:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")
#     elif sma20[i - 1] < sma100[i - 1] and sma20[i] > sma100[i]:
#         if rsi[i] < 70:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
#         else:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")
#     elif sma20[i - 1] > sma100[i - 1] and sma20[i] < sma100[i]:
#         if rsi[i] < 70:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
#         else:
#             ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")


# plt.show()
