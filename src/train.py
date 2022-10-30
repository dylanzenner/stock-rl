import datetime as dt
import numpy as np
import pandas_ta as ta
import pandas as pd
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from trading_env import StockTradingEnv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime


class SmartTrader:
    def __init__(self, api_key, secret_key, endpoint):
        self.api_key = api_key
        self.secret_key = secret_key
        self.endpoint = endpoint

    def get_data(self, symbol, timeframe, start_date, end_date):
        stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )
        bars = stock_client.get_stock_bars(request_params)
        df = bars.df.reset_index(level=["symbol", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df = df.drop(columns=["timestamp"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date")
        df["rsi"] = df.ta.rsi(
            close=df["close"], length=14, scalar=None, drift=None, offset=None
        )
        df["sma"] = df.ta.sma(
            close=df["close"], length=20, scalar=None, drift=None, offset=None
        )
        df["obv"] = df.ta.obv(
            close=df["close"], volume=df["volume"], scalar=None, drift=None, offset=None
        )
        df.fillna(0, inplace=True)

        return df

    def train_and_run_model(self, df, model, total_timesteps, log_interval, ensemble):
        if ensemble == True:
            pass
        else:
            if model in (PPO, A2C):
                if model == PPO:
                    name_var = "PPO"
                else:
                    name_var = "A2C"
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                model = model("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
                model.save(
                    "trained_model_{}_{}".format(
                        name_var, datetime.now().strftime("%d-%m-%y")
                    )
                )

                obs = env.reset()
                for i in range(20):
                    print("---------------------------")
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render()

            else:
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                n_actions = env.action_space.shape[-1]
                action_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
                model = model("MlpPolicy", env, action_noise=action_noise, verbose=1)
                model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
                model.save(
                    "trained_model_TD3_{}".format(datetime.now().strftime("%d-%m-%y"))
                )

                obs = env.reset()
                for i in range(4):
                    print("---------------------------")
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render()


if __name__ == "__main__":
    # initiate the class
    trader = SmartTrader(
        api_key="PK04UHV69AF2QULV4REU",
        secret_key="7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd",
        endpoint="https://paper-api.alpaca.markets",
    )

    # get the data
    df = trader.get_data(
        symbol="NVDA",
        timeframe=TimeFrame.Day,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1),
    )

    # train and run the model
    trader.train_and_run_model(
        df=df, model=TD3, total_timesteps=1, log_interval=10, ensemble=False
    )
