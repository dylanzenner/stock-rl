# this will be similar to the train.py file except we will import the live data instead of the historical data 
# so we will need to use web sockets
# keys required
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import StockTradingEnv
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
from stable_baselines3 import PPO
import pandas as pd
import pandas_ta as ta
import talib as TA
import numpy as np

# just toying around with how to stream in the stock data. Will have to figure out how to append the RSI values to the responses 
# and then pass it to the model for predictions
data = []
close_values = []
volume_values = []

async def print_crypto_trade(q):
    # loop through the item and put them in a dataframe
    
    close_values.append(q.close)
    volume_values.append(q.volume)
    data.append({
        'close': q.close, 'high': q.high,
        'low': q.low, 'open': q.open,
        'symbol': q.symbol, 'timestamp': q.timestamp,
        'trade_count': q.trade_count, 'volume': q.volume,
        'vwap': q.vwap, 'rsi': TA.RSI(np.array(close_values), 14)[-1],
        'sma': TA.SMA(np.array(close_values), 12)[-1],
        'obv': TA.OBV(np.array(close_values), np.array(volume_values))[-1]
    
    })
    df = pd.DataFrame(data)
    print(close_values)
    print(len(close_values))
    
    print(df)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=1, log_interval=10)
    obs = env.reset()
    for i in range(len(df)):
        print('---------------------------')
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


# Initiate Class Instance
stream = Stream('PK04UHV69AF2QULV4REU',
                '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd',
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# subscribing to event
stream.subscribe_crypto_bars(print_crypto_trade, 'BTCUSD')


# env = DummyVecEnv([lambda: StockTradingEnv(df)])


stream.run()



# somehow load the model and get it to work on the live bar data here