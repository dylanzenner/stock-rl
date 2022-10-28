import datetime as dt
import numpy as np
import pandas_ta as ta
import pandas as pd
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from trading_env import StockTradingEnv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# will have to hide these for production
key_id = 'PK04UHV69AF2QULV4REU'
secret_key = '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd'
endpoint = 'https://paper-api.alpaca.markets'

stock_client = StockHistoricalDataClient(key_id, secret_key)

# get 1 years worth data on Nvidia by the day. Could also use TimeFrame.Day for daily data
request_params = StockBarsRequest(symbol_or_symbols=['NVDA'],
                                  timeframe=TimeFrame.Day,
                                  start=datetime(2020, 1, 1),
                                  end=datetime(2021, 1, 1)
                                  )

bars = stock_client.get_stock_bars(request_params)

# convert to pandas dataframe
df = bars.df.reset_index(level=['symbol', 'timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['new_date_column'] = df['timestamp'].dt.date
df = df.drop(columns=['timestamp'])
df['new_date_column'] = pd.to_datetime(df['new_date_column'])
df.set_index('new_date_column')
df['rsi'] = df.ta.rsi(close=df['close'], length=14, scalar=None, drift=None, offset=None)
df['sma'] = df.ta.sma(close=df['close'], length=12, scalar=None, drift=None, offset=None)
df['obv'] = df.ta.obv(close=df['close'], volume=df['volume'], scalar=None, drift=None, offset=None)
df.fillna(0, inplace=True)
print(df.columns)

# # Set up the environment
env = DummyVecEnv([lambda: StockTradingEnv(df)])

# print(env.action_space)

# # # The noise objects for TD3
# # n_actions = env.action_space.shape[-1]
# # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


# # # train the model
model = PPO("MlpPolicy", env, verbose=1)
# # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

# # # Visualize how the model performs
obs = env.reset()
for i in range(2000):
    print('---------------------------')
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

# # # save the model so we can call it from our main file which will be our entry point for our bot
# # model.save('trained_model')

