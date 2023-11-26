import pandas as pd
from utilFunction import generateCsv
from constant import DEFAULT_STOCK_LIST
INITIAL_FUND = 25000

def doBackTest(stockNumber):
    resultArr = [['Date','Cash','Number of Stock', 'Stock Price', 'Total value']]
    filePath = "./source_data/" + stockNumber + ".csv"
    targetStockData = pd.read_csv(filePath)
    filePath = "./action/" + stockNumber + "_trade_decision" + ".csv"
    targetStockAction = pd.read_csv(filePath)
    cashAmount = INITIAL_FUND
    numberOfHoldingStock = 0
    for index , row in targetStockAction.iterrows():
        selected_rows = targetStockData[targetStockData["Date"] == row["Date"]]
        closePrice = selected_rows["Close"].item()
        if(row["Action"]=="BUY"):
            if(closePrice > cashAmount):
                #nothing do
                a=1
            else:
                price = closePrice
                numberOfStockPendingToBuy = cashAmount // price
                numberOfHoldingStock = numberOfHoldingStock + numberOfStockPendingToBuy
                cashAmount = cashAmount - (numberOfStockPendingToBuy * price)
                resultArr.append([row["Date"], cashAmount, numberOfHoldingStock,closePrice, cashAmount+numberOfHoldingStock*closePrice])
        elif(row["Action"]=="SELL"):
            if(0 >= numberOfHoldingStock):
                #nothing do
                a=1
            else:
                price = closePrice
                soldCash = numberOfHoldingStock *price
                cashAmount = cashAmount + soldCash
                numberOfHoldingStock = 0
                resultArr.append([row["Date"], cashAmount, numberOfHoldingStock, closePrice, cashAmount+numberOfHoldingStock*closePrice])
    generateCsv("./back_test_data/" + stockNumber + "_back_test.csv",resultArr)

def main():
    for stockNumber in DEFAULT_STOCK_LIST:
        doBackTest(stockNumber)

if __name__ == "__main__":
    main()
