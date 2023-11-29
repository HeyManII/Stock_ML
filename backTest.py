import pandas as pd
from utilFunction import generateCsv
from constant import DEFAULT_STOCK_LIST
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

INITIAL_FUND = 25000
START_DATE = "2022-11-01"
END_DATE = "2023-10-31"

model_used = "TA" # "TA" or "ML"
short_SMA_day = 10
long_SMA_day = 20

def doBackTest(stockNumber):
    resultArr = [["Date", "Cash", "Number of Stock", "Stock Price", "Total value"]]
    BuyDate = []
    BuyPrice = []
    SellDate = []
    SellPrice = []
    RSIArr = []
    short_SMA_Arr = []
    long_SMA_Arr = []
    BuySMA = []
    SellSMA = []
    if model_used == "TA":
        filePath = "./source_data/" + stockNumber + ".csv"
    elif model_used == "ML":
        filePath = "./source_data/ML" + stockNumber + ".csv"
    targetStockData = pd.read_csv(filePath)
    if model_used == "TA":
        filePath = "./action/" + stockNumber + "_trade_decision" + ".csv"
    elif model_used == "ML":
        filePath = "./action/" + stockNumber + "_MLtrade_decision" + ".csv"
    targetStockAction = pd.read_csv(filePath)
    if model_used == "TA":
        filePath = "./ta_data/" + stockNumber + "_calculated" + ".csv"
    CalculatedStockData = pd.read_csv(filePath)
    cashAmount = INITIAL_FUND
    numberOfHoldingStock = 0
    for index, row in targetStockData.iterrows():
        if datetime.strptime(row["Date"], '%Y-%m-%d')>=datetime.strptime(START_DATE, '%Y-%m-%d') and\
            datetime.strptime(row["Date"], '%Y-%m-%d')<=datetime.strptime(END_DATE, '%Y-%m-%d'):
            closePrice = row["Close"]
            TA_calculated = CalculatedStockData[CalculatedStockData["Date"] == row["Date"]]
            RSIArr.append(TA_calculated["RSI"].item())
            short_SMA_Arr.append(TA_calculated["SMA"+str(short_SMA_day)].item())
            long_SMA_Arr.append(TA_calculated["SMA"+str(long_SMA_day)].item())
            if targetStockAction["Date"].eq(row["Date"]).any():
                    trade_action = targetStockAction[targetStockAction["Date"] == row["Date"]]["Action"].item()
                    if trade_action == "BUY":
                        if closePrice > cashAmount:
                            resultArr.append(
                                [
                                    row["Date"],
                                    cashAmount,
                                    numberOfHoldingStock,
                                    closePrice,
                                    cashAmount + numberOfHoldingStock * closePrice,
                                            ]
                            )
                        else:
                            price = closePrice
                            numberOfStockPendingToBuy = cashAmount // price
                            numberOfHoldingStock = numberOfHoldingStock + numberOfStockPendingToBuy
                            cashAmount = cashAmount - (numberOfStockPendingToBuy * price)
                            resultArr.append(
                                [
                                    row["Date"],
                                    cashAmount,
                                    numberOfHoldingStock,
                                    closePrice,
                                    cashAmount + numberOfHoldingStock * closePrice,
                                ]
                            )
                            BuyDate.append(row["Date"])
                            BuyPrice.append(closePrice)
                            if model_used == "TA":
                                BuySMA.append(TA_calculated["SMA"+str(short_SMA_day)].item())

                    elif trade_action == "SELL":
                        if 0 >= numberOfHoldingStock:
                            # nothing do
                             resultArr.append(
                                [
                                    row["Date"],
                                    cashAmount,
                                    numberOfHoldingStock,
                                    closePrice,
                                    cashAmount + numberOfHoldingStock * closePrice,
                                            ]
                )
                        else:
                            price = closePrice
                            soldCash = numberOfHoldingStock * price
                            cashAmount = cashAmount + soldCash
                            numberOfHoldingStock = 0
                            resultArr.append(
                                [
                                    row["Date"],
                                    cashAmount,
                                    numberOfHoldingStock,
                                    closePrice,
                                    cashAmount + numberOfHoldingStock * closePrice,
                                ]
                            )
                            SellDate.append(row["Date"])
                            SellPrice.append(closePrice)
                            if model_used == "TA":
                                SellSMA.append(TA_calculated["SMA"+str(short_SMA_day)].item())
            else:
                resultArr.append(
                    [
                        row["Date"],
                        cashAmount,
                        numberOfHoldingStock,
                        closePrice,
                        cashAmount + numberOfHoldingStock * closePrice,
                                ]
                )
    if model_used =="TA":
        generateCsv("./back_test_data/" + stockNumber + "_back_test.csv", resultArr)
    elif model_used =="ML":
        generateCsv("./back_test_data/" + stockNumber + "_ML_back_test.csv", resultArr)

    # Plot graph
    graph_date = []
    graph_stock = []
    graph_total = []
    for i in range(len(resultArr)):
        if i>0:
            graph_date.append(resultArr[i][0])
            graph_stock.append(resultArr[i][3])
            graph_total.append(resultArr[i][4])
    graph_date = np.asarray(graph_date)
    graph_stock = np.asarray(graph_stock)
    graph_total = np.asarray(graph_total)
    BuyDate = np.asarray(BuyDate)
    BuyPrice = np.asarray(BuyPrice)
    SellDate = np.asarray(SellDate)
    SellPrice = np.asarray(SellPrice)
    RSIArr = np.asarray(RSIArr)
    short_SMA_Arr = np.asarray(short_SMA_Arr)
    long_SMA_Arr = np.asarray(long_SMA_Arr)
    SellSMA = np.asarray(SellSMA)
    BuySMA = np.asarray(BuySMA)
    
    fig1,ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Today Close', color=color)
    ax1.plot(graph_date, graph_stock, color=color,zorder=1)
    ax1.scatter(BuyDate,BuyPrice, marker="^",zorder=2,color='0')
    ax1.scatter(SellDate,SellPrice, marker="v",zorder=2,color='0')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set(xticks=graph_date[::60])

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Value', color=color)  
    ax2.plot(graph_date, graph_total, color=color,zorder=1)
    ax2.tick_params(axis='y', labelcolor=color)

    fig1.tight_layout() 
    #plt.show()
    if model_used =="TA":
        fig1.savefig("./back_test_data/" + stockNumber + "_back_test.png")
    elif model_used =="ML":
        fig1.savefig("./back_test_data/ML" + stockNumber + "_back_test.png")
    
    if model_used =="TA":
        fig2, (ax3, ax4) = plt.subplots(nrows=2, sharex=True)
        color = '0'
        ax3.set_xlabel('Date')
        ax3.set_ylabel('SMA', color=color)
        ax3.plot(graph_date, short_SMA_Arr, color='tab:blue',zorder=1,label='SMA'+str(short_SMA_day))
        ax3.plot(graph_date, long_SMA_Arr, color='tab:green',zorder=1,label='SMA'+str(long_SMA_day))
        ax3.scatter(BuyDate,BuySMA, marker="^",zorder=2,color='0')
        ax3.scatter(SellDate,SellSMA, marker="v",zorder=2,color='0')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set(xticks=graph_date[::60])
        ax3.legend()

        ax4.plot(graph_date, RSIArr, color="0",zorder=1)
        ax4.axhline(70,color="tab:red")
        ax4.axhline(30,color="tab:red")
        
        for i,row in enumerate(BuyDate): 
            ax3.axvline(BuyDate[i], color='tab:grey', linestyle='--')
            ax4.axvline(BuyDate[i], color='tab:grey', linestyle='--')
        
        for i,row in enumerate(SellDate): 
            ax3.axvline(SellDate[i], color='tab:grey', linestyle=':')
            ax4.axvline(SellDate[i], color='tab:grey', linestyle=':')
        
        fig2.savefig("./back_test_data/" + stockNumber + "_RSI SMA.png")

        
def main():
    for stockNumber in DEFAULT_STOCK_LIST:
        doBackTest(stockNumber)



if __name__ == "__main__":
    main()
