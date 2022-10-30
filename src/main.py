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


# Initiate Class Instance
stream = Stream('PK04UHV69AF2QULV4REU',
                '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd',
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription


env = DummyVecEnv([lambda: StockTradingEnv(stream)])
model = PPO("MlpPolicy", env, verbose=1)
# # # # # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100, log_interval=10)

# # # # Visualize how the model performs
obs = env.reset()
# print(obs)
while True:
    print('---------------------------')
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()


# env = DummyVecEnv([lambda: StockTradingEnv(df)])


stream.run()



# somehow load the model and get it to work on the live bar data here