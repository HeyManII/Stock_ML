import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

if os.path.isfile("./qqq2010.csv"):
    qqq = pd.read_csv("./qqq2010.csv")
else:
    # Download QQQ data from 2010 to 2022
    print("downloading data")
    qqq = yf.download("QQQ", start="2010-01-01", end="2010-12-31")

# Print the first 5 rows of the data
print(qqq.head())

# Download QQQ data from 2010 to 2022
qqq = yf.download("QQQ", start="2010-01-01", end="2010-12-31")

# Print the first 5 rows of the data
print(qqq.head())

fig = plt.figure(figsize=(10, 5))
plt.plot(qqq["Open"], color="red")
plt.title("Google open price")
plt.show()

fig = plt.figure(figsize=(10, 5))
plt.plot(qqq["Open"], color="red")
plt.plot(qqq["Close"], color="green")
plt.title("Google open price")
plt.legend()
plt.show()

qqq.to_csv("./qqq2010.csv")
