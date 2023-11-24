import pandas as pd
import csv

INITIAL_FUND = 25000

def doBackTest(stockNumber):
    resultArr = [['Date','Cash','Number of Stock', 'Stock Price', 'Total value']]
    filePath = "./" + stockNumber + ".csv"
    targetStockData = pd.read_csv(filePath)
    filePath = "./" + stockNumber + "_trade_decision" + ".csv"
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
            print("sell")
            if(0 >= numberOfHoldingStock):
                #nothing do
                a=1
            else:
                price = closePrice
                soldCash = numberOfHoldingStock *price
                cashAmount = cashAmount + soldCash
                numberOfHoldingStock = 0
                resultArr.append([row["Date"], cashAmount, numberOfHoldingStock, closePrice, cashAmount+numberOfHoldingStock*closePrice])
    generateCsv(stockNumber+"_back_test.csv",resultArr)


def generateCsv(fileName, data):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write each row of data to the csv file
        for row in data:
            writer.writerow(row)

def tradeTrigger(stockNumber , originStockData):
    #stragey 2 hit 1 >> buy
    #stragey 2 hit 1 >> sell
    tradeArr =[['Date','Action']]
    for index, row in originStockData.iterrows():
        # RSI Buy Stock Flag	RSI Sell Stock Flag	SMA Buy Stock Flag	SMA Sell Stock Flag
        if((row['RSI Buy Stock Flag'] == 'True' or row['SMA Buy Stock Flag'] == 'True' )):
            tradeArr.append([row['Date'],'BUY'])
        elif((row['RSI Sell Stock Flag'] == 'True' or row['SMA Sell Stock Flag'] == 'True' )):
            tradeArr.append([row['Date'],'SELL'])  
    generateCsv(stockNumber+"_trade_decision.csv",tradeArr)
    doBackTest(stockNumber)  

def main():
    doBackTest("2800")

if __name__ == "__main__":
    main()