# Step 0: Install Library

> pip install yfinance matplotlib panda

# Step 1: start main

> py main.py

# Strategy

RSI >=70 >> sell
SMA20 < SMA100 >> sell

RSI <=30 >> buy
SMA20 > SMA100 >> buy
