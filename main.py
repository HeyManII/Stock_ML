import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

if os.path.isfile("./2800.csv"):
    hk_stock_2800 = pd.read_csv("./2800.csv")
else:
    # Download QQQ data from 2010 to 2022
    print("downloading data")
    hk_stock_2800 = yf.download("2800.HK", start="2015-01-01", end="2023-05-31")
    hk_stock_2800.to_csv("./2800.csv", index=False)

# Print the first 5 rows of the data
# print(hk_stock_2800.head())

hk_stock_2800["SMA20"] = hk_stock_2800["Close"].rolling(window=20).mean()
hk_stock_2800["SMA100"] = hk_stock_2800["Close"].rolling(window=100).mean()
delta = hk_stock_2800["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# ax[0].plot(hk_stock_2800["Open"], color="red", label="Open")
ax[0].plot(hk_stock_2800["Close"], color="green", label="Close")
ax[0].plot(hk_stock_2800["SMA20"], color="blue", label="SMA20")
ax[0].plot(hk_stock_2800["SMA100"], color="orange", label="SMA100")
ax[0].set_title("2800 open price")
ax[0].legend()

ax[1].plot(hk_stock_2800.index, rsi, label="RSI")
ax[1].axhline(y=30, color="red", linestyle="--")
ax[1].axhline(y=70, color="red", linestyle="--")
ax[1].set_title("2800 RSI")
ax[1].legend()


# add red circle when SMA20 = SMA100
sma20 = hk_stock_2800["SMA20"]
sma100 = hk_stock_2800["SMA100"]
for i in range(1, len(sma20)):
    if sma20[i] == sma100[i]:
        if rsi[i] < 70:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
        else:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")
    elif sma20[i - 1] < sma100[i - 1] and sma20[i] > sma100[i]:
        if rsi[i] < 70:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
        else:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")
    elif sma20[i - 1] > sma100[i - 1] and sma20[i] < sma100[i]:
        if rsi[i] < 70:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="red")
        else:
            ax[0].plot(hk_stock_2800.index[i], sma20[i], "o", color="black")


plt.show()
