# this will be similar to the train.py file except we will import the live data instead of the historical data 
# so we will need to use web sockets
# keys required
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.a2c import MlpPolicy

from trading_env import StockTradingEnv
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
from stable_baselines3 import PPO
import pandas as pd
import pandas_ta as ta
import talib as TA
import numpy as np
import time
import alpaca_trade_api as tradeapi

api = tradeapi.REST(key_id='PK04UHV69AF2QULV4REU', secret_key='7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd', base_url=URL('https://paper-api.alpaca.markets'))
account = api.get_account()
print(account)

available_cash = account.cash

data = []
close_values = []
volume_values = []
df = None
async def print_crypto_trade(q):
    global df, close_values, volume_values, data
    print('crypto', q)
    # loop through the item and put them in a dataframe

    close_values.append(q.close)
    volume_values.append(q.volume)
    data.append({
        'open': q.open,
        'close': q.close,
        'rsi': TA.RSI(np.array(close_values), 14)[-1],
        'sma': TA.SMA(np.array(close_values), 12)[-1],
        'obv': TA.OBV(np.array(close_values), np.array(volume_values))[-1]

    })
    df = pd.DataFrame(data)
    print('columns: {}'.format(df.columns))
    df.fillna(0, inplace=True)
    print(close_values)
    print(len(close_values))
    model = PPO.load("trained_model_PPO_30-10-22")
    if len(data) >= 14:
        action = model.predict(df.tail(6))
        print('ACTION: {}'.format(action[0][0]))
        if action < 0:
            # number of stock to sell
            sell_off_stock = int(account.cash) * 0.1 / q.close
            
            api.submit_order(
                symbol='NVDA',
                qty=sell_off_stock,  # fractional shares
                side='sell',
                type='market',
                time_in_force='day',
            )

        if action > 0:
            api.submit_order(
                symbol='SPY',
                qty=1.5,  # fractional shares
                side='buy',
                type='market',
                time_in_force='day',
            )
        




# Initiate Class Instance
stream = Stream('PK04UHV69AF2QULV4REU',
                '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd',
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription



stream.subscribe_crypto_bars(print_crypto_trade, 'BTCUSD')




stream.run()




# somehow load the model and get it to work on the live bar data here