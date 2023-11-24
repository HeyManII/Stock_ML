import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from backTest import doBackTest, tradeTrigger

SHORT_SMA_DAY = 20
LONG_SMA = 100
RSI_DAY = 14
RSI_SELL_INDEX = 70
RSI_BUY_INDEX = 30
INITIAL_AMOUNT_STOCK = 1


def getStockData(stockNumber, startTime, endTime):
    print("stock number ", stockNumber)
    filePath = "./" + stockNumber + ".csv"
    stockData = pd.read_csv(filePath)
    return stockData

def calculateSmaInStockData(originStockData, day, referenceColumnName):
    newStockData = originStockData
    newStockData["SMA" + str(day)] = (
        newStockData[referenceColumnName].rolling(window=day).mean()
    )
    return newStockData


def createRsiInStockData(originStockData, day):
    newStockData = originStockData
    delta = newStockData["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=day).mean()
    avg_loss = loss.rolling(window=day).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    newStockData["RSI"] = rsi
    return newStockData


def calculateRsiStrategy(originStockData):
    # row["Date"]
    initialBuyDone = False
    buyCount = 0
    sellCount = 0
    currentProfile = 0
    numberOfStockHold = 0
    for index, row in originStockData.iterrows():
        buyFlag = False
        sellFlag = False
        if math.isnan(row["RSI"]):
            a = 1
        else:
            previousValue = originStockData.loc[index - 1, "RSI"]
            currentValue = row["RSI"]
            if(math.isnan(previousValue)):
                if(currentValue<=RSI_BUY_INDEX):
                    buyFlag= True
                    # buyCount+=1
                    # currentProfile = currentProfile - row['Close']
                    # numberOfStockHold = numberOfStockHold+1
                    # initialBuyDone=True
                elif(currentValue >=RSI_SELL_INDEX):
                    # sell stock
                    sellFlag= True
                    # buyCount = buyCount-1
                    # currentProfile = currentProfile + row['Close']
                    # numberOfStockHold = numberOfStockHold-1
                    # initialBuyDone=True
            elif(currentValue >=RSI_SELL_INDEX):
                if(previousValue >=RSI_SELL_INDEX):
                    a=1
                else:
                    # sell stock
                    sellFlag= True
                    # buyCount = buyCount-1
                    # currentProfile = currentProfile + row['Close']
                    # numberOfStockHold = numberOfStockHold-1
                    # initialBuyDone=True
            elif(currentValue <=RSI_BUY_INDEX):
                if(previousValue <=RSI_BUY_INDEX):
                    a=1
                else:
                    # buy stock
                    buyFlag= True
                    # sellCount+=1
                    # currentProfile = currentProfile - row['Close']
                    # numberOfStockHold = numberOfStockHcurrentProfileold+1   
        originStockData.loc[index,"RSI Buy Stock Flag"]= str(buyFlag)
        originStockData.loc[index,"RSI Sell Stock Flag"]= str(sellFlag)
    return originStockData


def calculateSmaStrategy(originStockData, shorterSma, longerSma):
    initialBuyDone = False
    buyCount = 0
    sellCount = 0
    currentProfile = 0
    numberOfStockHold = 0
    # for index, row in enumerate(originStockData.iterrows()):
    for index, row in originStockData.iterrows():
        buyFlag = False
        sellFlag = False
        if math.isnan(row[longerSma]):
            a = 1
        else:
            # shortMa > longMa >> buy
            # longMa > shortMa >> sell
            row[shorterSma]
            row[longerSma]
            previousShort = originStockData.loc[index - 1, shorterSma]
            previousLong = originStockData.loc[index - 1, longerSma]
            if (row[shorterSma] - row[longerSma]) > 0:
                if (previousShort - previousLong) > 0:
                    a = 1
                else:
                    #buy stock
                    buyFlag = True
                    # sellCount+=1
                    # currentProfile = currentProfile - row['Close']
                    # numberOfStockHold = numberOfStockHold+1   
            elif((row[longerSma] - row[shorterSma]) >0):
                if((previousLong - previousShort)>0):
                    a=1
                else:
                    #sell stock
                    sellFlag = True
                    # buyCount = buyCount-1
                    # currentProfile = currentProfile + row['Close']
                    # numberOfStockHold = numberOfStockHold-1  
        originStockData.loc[index,"SMA Buy Stock Flag"]= str(buyFlag)
        originStockData.loc[index,"SMA Sell Stock Flag"]= str(sellFlag)
    return originStockData


def generateParticularStockDataWithDiagram(stockNumber, startTime, endTime):
    taItems = ["SMA", "RSI"]
    stockData = getStockData(stockNumber, startTime, endTime)
    stockData = calculateSmaInStockData(stockData, SHORT_SMA_DAY, "Close")
    stockData = calculateSmaInStockData(stockData, LONG_SMA, "Close")
    stockData = createRsiInStockData(stockData, RSI_DAY)
    stockData = calculateRsiStrategy(stockData)
    stockData = calculateSmaStrategy(stockData,"SMA"+str(SHORT_SMA_DAY),"SMA"+str(LONG_SMA))
    tradeTrigger(stockNumber,stockData)
    # stockData.to_csv("./"+stockNumber+"_calculated"+".csv", index=True)
    
    # row = len(taItems)
    # column = 1
    # fig, ax = plt.subplots(row, column, figsize=(10, 10), sharex=True)
    # for index, taItem in enumerate(taItems):
    #     if taItem == "SMA":
    #         ax[index].plot(stockData["Close"], color="green", label="Close")
    #         ax[index].plot(stockData["SMA"+str(SHORT_SMA_DAY)], color="blue", label="SMA"+str(SHORT_SMA_DAY))
    #         ax[index].plot(stockData["SMA"+str(LONG_SMA)], color="orange", label="SMA"+str(LONG_SMA))
    #         ax[index].set_title(stockNumber+" open price")
    #         ax[index].legend()
    #     elif taItem == "RSI":        
    #         ax[index].plot(stockData["RSI"], color="green", label="RSI")
    #         ax[index].axhline(y=RSI_BUY_INDEX, color="red", linestyle="--")
    #         ax[index].axhline(y=RSI_SELL_INDEX, color="red", linestyle="--")
    #         ax[index].set_title(stockNumber+" RSI")
    #         ax[index].legend()
    
    # plt.show()
    stockData = calculateSmaStrategy(
        stockData, "SMA" + str(SHORT_SMA_DAY), "SMA" + str(LONG_SMA)
    )
    stockData.to_csv("./" + stockNumber + "_calculated" + ".csv", index=False)
    row = len(taItems)
    column = 1
    fig, ax = plt.subplots(row, column, figsize=(10, 10), sharex=True)
    for index, taItem in enumerate(taItems):
        if taItem == "SMA":
            ax[index].plot(stockData["Close"], color="green", label="Close")
            ax[index].plot(
                stockData["SMA" + str(SHORT_SMA_DAY)],
                color="blue",
                label="SMA" + str(SHORT_SMA_DAY),
            )
            ax[index].plot(
                stockData["SMA" + str(LONG_SMA)],
                color="orange",
                label="SMA" + str(LONG_SMA),
            )
            ax[index].set_title(stockNumber + " open price")
            ax[index].legend()
        elif taItem == "RSI":
            ax[index].plot(stockData["RSI"], color="green", label="RSI")
            ax[index].axhline(y=RSI_BUY_INDEX, color="red", linestyle="--")
            ax[index].axhline(y=RSI_SELL_INDEX, color="red", linestyle="--")
            ax[index].set_title(stockNumber + " RSI")
            ax[index].legend()

    plt.show()


def findBuySignal(stockData):
    # initial the array to store buy in signal date and price
    buySignal = []
    # if SMA20 cross SMA100 from below to above && RSI < 70, BUY
    for index, row in stockData.iterrows():
        if row["SMA20"] >= row["SMA100"]:
            prev_row = stockData.iloc[index - 1]
            if prev_row["SMA20"] < prev_row["SMA100"]:
                buySignal.append([index + 1, "Buy", row["Close"]])
    return buySignal


def findSellSignal(stockData):
    # initial the array to store buy in signal date and price
    sellSignal = []
    for index, row in stockData.iterrows():
        # if SMA20 cross SMA100 from above to below, SELL
        if row["SMA20"] <= row["SMA100"]:
            prev_row = stockData.iloc[index - 1]
            if prev_row["SMA20"] > prev_row["SMA100"]:
                sellSignal.append([index + 1, "Sell", row["Close"]])
    return sellSignal


def trade(capital, buySignal, sellSignal):
    stockAmount = 0
    cash = capital
    # Combine buySignal and sellSignal
    signals = buySignal + sellSignal
    # Sort signals according to column 0
    signals.sort(key=lambda x: x[0])
    signals = pd.DataFrame(signals)
    signals[2] = signals[2].round(1)
    for index, row in signals.iterrows():
        if row[1] == "Buy":
            amountToBuy = math.floor((cash / (row[2] * 100)))
            stockAmount = stockAmount + amountToBuy * 100
            cash = cash - row[2] * amountToBuy * 100
            # print(f"Bought {amountToBuy*100} shares at {row[2]} per share")
            # print(f"Cash left:{cash}")
        if row[1] == "Sell":
            # print(f"Sell {stockAmount} shares at {row[2]} per share")
            cash = cash + row[2] * stockAmount
            stockAmount = 0
            # print(f"Cash left:{cash}")
    return cash, stockAmount


def main():
    startTime = "2015-01-01"
    endTime = "2023-05-31"
    generateParticularStockDataWithDiagram("2800", startTime, endTime)
    generateParticularStockDataWithDiagram("0700", startTime, endTime)
    # stock_700 = pd.read_csv("./0700_calculated.csv")
    # buySignal = findBuySignal(stock_700)
    # sellSignal = findSellSignal(stock_700)

    # capital = 100000
    # cash, stockAmount = trade(capital, buySignal, sellSignal)
    # print("cash:", cash)
    # print("stockAmount:", stockAmount)


if __name__ == "__main__":
    main()


# # step1 pick stock number

# # step2 create csv file
####################################################### original code
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
