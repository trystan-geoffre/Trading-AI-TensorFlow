import yfinance as yf
import pandas as pd
import os

# Read stock symbol listed in the S&P from Wikipedia
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Define the date range for the last 5 years
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=5)

# Create a directory to store historical data
data_dir = 'sp500_data'
os.makedirs(data_dir, exist_ok=True)

# Download historical data for each S&P 500 company
for ticker in sp500_tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(os.path.join(data_dir, f'{ticker}.csv'))