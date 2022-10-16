import gym
import json
import datetime as dt
import numpy as np

from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from trading_env import StockTradingEnv

import pandas as pd

# we should update the data everyday when we spin up our bot
# so instead of reading from a csv we will just get the updated data
# from the api and put it in a dataframe. That way we have an up to date
# model everyday.

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# will have to hide these for production
key_id = 'PK04UHV69AF2QULV4REU'
secret_key = '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd'
endpoint = 'https://paper-api.alpaca.markets'

stock_client = StockHistoricalDataClient(key_id, secret_key)

# get 1 years worth data on Nvidia by the day. Could also use TimeFrame.Minute for daily data
request_params = StockBarsRequest(symbol_or_symbols=['NVDA'],
                                  timeframe=TimeFrame.Minute,
                                  start=datetime(2020, 1, 1),
                                  end=datetime(2021, 1, 1)
                                  )

bars = stock_client.get_stock_bars(request_params)
# should probably define all of the stocks we will be using in an array and then
# aggregate them all into a single dataframe so they are all in one place for our algorithm
# but should try our algorithm with just 1 stock until we know it works

df = bars.df.reset_index(level=['symbol', 'timestamp'])

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['new_date_column'] = df['timestamp'].dt.date
df = df.drop(columns=['timestamp'])
df['new_date_column'] = pd.to_datetime(df['new_date_column'])
df.set_index('new_date_column')
df = df.rename(columns={'close': 'Close', 'symbol': 'Symbol', 'open':'Open', 'high':'High', 'low':'Low', 'volume':'Volume', 'trade_count': 'Trade_Count'})
print(df)

# add the RSI value to the dataframe


df = df.sort_index()

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])


model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

# Visualize how the model performs
obs = env.reset()
for i in range(2000):
    print('---------------------------')
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

# save the model so we can call it from our main file which will be our entry point for our bot
model.save('trained_model')

