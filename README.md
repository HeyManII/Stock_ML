## Step 0: Install Library

> pip install yfinance matplotlib panda scikit-learn

## Step 1: Setting a portfolio

The file "Porfolio.py" chooses the 4 stocks with highest masket cap in 4 HSI sectors respectively.
By Runningf the code,

> py Portfolio.py

it shows which 4 stocks are chosen and calculate the optimal weighting of the 4 stocks in the portofolio.
The portfolio is set to be the most conservative. (with the lowest std)

## Step 1.1: First Init problem

py getData.py

## Step 2: start main

> py main.py

# Strategy

Sell:

- RSI >=70
- SMA20 < SMA100

Buy:

- RSI <=30
- SMA20 > SMA100
