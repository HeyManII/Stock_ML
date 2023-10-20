import yfinance as yf

# Download QQQ data from 2010 to 2022
qqq = yf.download("QQQ", start="2010-01-01", end="2010-12-31")

# Print the first 5 rows of the data
print(qqq.head())

