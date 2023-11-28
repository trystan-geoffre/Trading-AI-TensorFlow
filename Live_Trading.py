# Imports
import tensorflow as tf
import yfinance as yf
import pandas as pd
import schedule
import time
import pytz
import talib as ta
from datetime import datetime, timedelta
import retrying
import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from config import *
from backtesting import Backtest
from talib import EMA, SMA

# Define your Alpaca API keys here
ALPACA_API_KEY = API_KEY
ALPACA_SECRET_KEY = SECRET_KEY

# Connection to Alpaca API
def connect(api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    try:
        account = api.get_account()
        print("Connected: Connected to Alpaca API")
        return api
    except Exception as e:
        print(f"Failed to connect to Alpaca API. Error: {str(e)}")
        return None

api_key = API_KEY
api_secret = SECRET_KEY

api = connect(api_key, api_secret)

# Normalize Function
def normalize_series(data, min_value, max_value):
    data = (data - min_value) / (max_value - min_value)
    return data

# Denormalize Function
def denormalize_series(normalized_data, min_value, max_value):
    original_data = normalized_data * (max_value - min_value) + min_value
    return original_data

# Windowed Dataset Function
def windowed_dataset(series,batch_size , n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

# Fetching Stock Data Function
def get_data(pair):

    start_date = datetime.now(pytz.timezone('GTC/Paris')) - timedelta(days=365)
    end_date = datetime.now(pytz.timezone('GTC/Paris'))

    stock_data = yf.download(pair, start=start_date, end=end_date)

    data = pd.DataFrame({'Date': stock_data.index, 'Open': stock_data['Open'], 'High': stock_data['High'],
                         'Low': stock_data['Low'], 'Close': stock_data['Close'], 'Volume': stock_data['Volume']})
    data.set_index('Date', inplace=True)

# Model Prediction Function
def model_predict(pair):
    # Load your pre-trained deep learning model
    model = tf.keras.models.load_model('mymodel.h5')

    # Load the CSV file and select the columns of interest
    df = get_data(pair)

    N_FEATURES = len(df.columns) 

    # Normalizes the data
    data = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = normalize_series(data, data_min, data_max)

    N_PAST = 10
    N_FUTURE = 10
    SHIFT = 1
    BATCH_SIZE = 128

    data = windowed_dataset(data, batch_size=BATCH_SIZE,
                            n_past=N_PAST, n_future=N_FUTURE,
                            shift=SHIFT)
    
    # Convert the data to a NumPy array of float32
    input_data = np.concatenate([x.numpy() for x, _ in data])

    # Make predictions using the loaded model
    predictions_normalized = model.predict(input_data)
    predictions = np.array([denormalize_series(predictions_normalized[:, i], data_min[i], data_max[i]) for i in range(N_FEATURES)]).T

    return predictions


# Trading Strategy Function
def MyStrategy(pair):
    # Get model predictions
    predictions = model_predict(pair)

    # Define the number of days for BUY and SELL signals
    days_to_buy = 10
    days_to_sell = 6

    # Initialize variables to keep track of consecutive days
    consecutive_up_days = 0
    consecutive_down_days = 0

    # Determine trading actions based on predictions
    for i in range(len(predictions)):
        if i + days_to_buy < len(predictions):
            # Check for consecutive days of price increase
            if all(predictions[i + j][0] > 0 for j in range(days_to_buy)):
                consecutive_up_days += 1
                if consecutive_up_days == days_to_buy:
                    # Place a BUY order
                    execute_trade(pair, "BUY", 1)
                    consecutive_up_days = 0  # Reset the counter
            else:
                consecutive_up_days = 0  # Reset the counter

        if i + days_to_sell < len(predictions):
            # Check for consecutive days of price decrease
            if all(predictions[i + j][0] < 0 for j in range(days_to_sell)):
                consecutive_down_days += 1
                if consecutive_down_days == days_to_sell:
                    # Place a SELL order
                    execute_trade(pair, "SELL", 1)
                    consecutive_down_days = 0  # Reset the counter
            else:
                consecutive_down_days = 0



# Backtesting Function
# Fetches historical data from Alpaca and runs a backtest using the defined strategy
def run_backtest(pair):
    start_date = datetime.now(pytz.timezone('GTC/Paris')) - timedelta(days=365)
    end_date = datetime.now(pytz.timezone('GTC/Paris'))

    # Fetch historical data from Alpaca
    stock_data = yf.download(pair, start=start_date, end=end_date)

    # Prepare data for Backtesting.py
    data = pd.DataFrame({'Date': stock_data.index, 'Open': stock_data['Open'], 'High': stock_data['High'],
                         'Low': stock_data['Low'], 'Close': stock_data['Close'], 'Volume': stock_data['Volume']})
    data.set_index('Date', inplace=True)

    # Run backtest
    bt = Backtest(data, MyStrategy)
    bt.run()


# Define a function to execute trades using Alpaca
@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def execute_trade(pair, order_type, size, tp_distance=None, stop_distance=None):
    # Check if the symbol is tradable
    asset = api.get_asset(pair)
    if not asset.tradable:
        print(f"{pair} is not tradable.")
        return

    print(f"{pair} found and is tradable!")

    if order_type == "BUY":
        order_action = "buy"
        price = api.get_latest_trade(pair).price
        if stop_distance:
            stop_price = price * (1 - stop_distance)
        if tp_distance:
            take_profit_price = price * (1 + tp_distance)

    elif order_type == "SELL":
        order_action = "sell"
        price = api.get_latest_trade(pair).price
        if stop_distance:
            stop_price = price * (1 - stop_distance)
        if tp_distance:
            take_profit_price = price * (1 + tp_distance)

    try:
        # Place the market order
        market_order = api.submit_order(
            symbol=pair,
            qty=size,
            side=order_action,
            type="market",
            time_in_force="gtc",
        )
        print("Market order successfully placed!")

        # Place the take profit limit order if tp_distance is specified
        if tp_distance:
            take_profit_price = round(take_profit_price, 2)
            tp_order = api.submit_order(
                symbol=pair,
                qty=size,
                side="sell" if order_action == "buy" else "buy",
                type="limit",
                time_in_force="gtc",
                limit_price=take_profit_price,
            )
            print("Take profit limit order successfully placed!")

        if stop_distance:
            stop_price = round(stop_price, 2)
            sl_order = api.submit_order(
                symbol=pair,
                qty=size,
                side="sell" if order_action == "buy" else "buy",
                type="stop_limit",
                time_in_force="gtc",
                stop_price=stop_price,
                limit_price=take_profit_price,  # Set limit price equal to stop price
            )
            print("Stop loss limit order successfully placed!")

    except Exception as e:
        print(f"Failed to send order: {str(e)}")


# Define a function to close a position by asset ID
def close_position(asset_id):
    try:
        api.close_position(asset_id)
        print("Order successfully closed!")
    except Exception as e:
        print(f"Failed to close order: {str(e)}")

# Define a function to close all positions for a given symbol
def close_positions_by_symbol(symbol):
    open_positions = positions_get()
    if not open_positions.empty:
        for idx, row in open_positions.iterrows():
            asset_id = row['_raw']['asset_id']
            close_position(asset_id)

def positions_get():
    positions = api.list_positions()
    if positions:
        df = pd.DataFrame([pos.__dict__ for pos in positions])
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        return df
    return pd.DataFrame()

# Live Trading and Schedule Setup
def live_trading():
    schedule.every().hour.at(":00").do(run_trader, '15Min')
    schedule.every().hour.at(":15").do(run_trader, '15Min')
    schedule.every().hour.at(":30").do(run_trader, '15Min')
    schedule.every().hour.at(":45").do(run_trader, '15Min')

    while True:
        schedule.run_pending()
        time.sleep(1)

# Run the strategy with the specific pairs
def run_trader(time_frame):
    print("Running trader at", datetime.now())
    pairs = ['AAPL', 'SPY', 'MSFT', 'META']  # Add your desired trading pairs here

    for pair in pairs:
        execute_trade(pair, "BUY", 1, 0.3, 0.3) 


# Main Execution
if __name__ == '__main__':
    live_trading()