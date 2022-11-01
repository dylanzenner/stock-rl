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
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
import pandas_ta as ta
import talib as TA
import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd


class SmartTrader:
    def __init__(self, api_key, secret_key, endpoint, api, stream=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.stream = stream
        self.models_to_ensemble = []
        self.close_values = []
        self.volume_values = []
        self.data = []
        self.api = api

    def get_data(self, symbol, timeframe, start_date, end_date):
        stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request_params = StockBarsRequest(
            symbol_or_symbols=[*symbol] if type(symbol) == type([]) else [symbol],
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
            close=df["close"],
            volume=df["volume"],
            scalar=None,
            drift=None,
            offset=None,
        )
        df.fillna(0, inplace=True)

        return df

    def train_and_run_model(
        self, df, model, total_timesteps, log_interval, ensemble, visualize=False
    ):
        if ensemble:
            # train the models
            for i in model:
                if i == PPO:
                    name_var = "PPO"
                elif i == TD3:
                    name_var = "TD3"
                else:
                    name_var = "A2C"
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                model_name = i("MlpPolicy", env, verbose=1)
                model_name.learn(
                    total_timesteps=total_timesteps, log_interval=log_interval
                )
                model_name.save(
                    "trained_model_{}_{}".format(
                        name_var, datetime.now().strftime("%d-%m-%y")
                    )
                )
                self.models_to_ensemble.append(
                    "trained_model_{}_{}".format(
                        name_var, datetime.now().strftime("%d-%m-%y")
                    )
                )

            if visualize:
                model_1 = model[0].load(self.models_to_ensemble[0])
                model_2 = model[1].load(self.models_to_ensemble[1])
                # model_3 = model[2].load(models_to_ensemble[2]) model for TD3

                obs = env.reset()
                for i in range(2000):
                    print("---------------------------")
                    action = model_1.predict(obs)[0] + model_2.predict(obs)[0] / 2
                    obs, rewards, done, info = env.step(action)
                env.render()

        else:
            if type(model) != type([]) and model in (PPO, A2C):
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

                if visualize:
                    obs = env.reset()
                    for i in range(2000):
                        print("---------------------------")
                        action, _states = model.predict(obs)
                        obs, rewards, done, info = env.step(action)
                    env.render()

            else:
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                n_actions = env.action_space.shape[-1]
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
                model = model("MlpPolicy", env, action_noise=action_noise, verbose=1)
                model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
                model.save(
                    "trained_model_TD3_{}".format(datetime.now().strftime("%d-%m-%y"))
                )

                if visualize:
                    obs = env.reset()
                    for i in range(4):
                        print("---------------------------")
                        action, _states = model.predict(obs)
                        print("Action: {}, States: {}".format(action, _states))
                        obs, rewards, done, info = env.step(action)
                    env.render()

    def run_model(self, model_names, ensemble=None, tickers=None):
        global model_1, model_2, model_3
        if ensemble is not None:
            for model in model_names:
                if "PPO" in model:
                    model_1 = PPO.load(model)
                elif "TD3" in model:
                    model_2 = TD3.load(model)
                else:
                    model_3 = A2C.load(model)

            async def print_crypto_trade(q):
                account = self.api.get_account()
                cash = float(account.cash)
                print(account)

                self.close_values.append(q.close)
                self.volume_values.append(q.volume)
                self.data.append(
                    {
                        "open": q.open,
                        "close": q.close,
                        "rsi": TA.RSI(np.array(self.close_values), 14)[-1],
                        "sma": TA.SMA(np.array(self.close_values), 12)[-1],
                        "obv": TA.OBV(
                            np.array(self.close_values), np.array(self.volume_values)
                        )[-1],
                    }
                )

                df = pd.DataFrame(self.data)
                print("columns: {}".format(df.columns))
                df.fillna(0, inplace=True)
                print(self.close_values)
                print(len(self.close_values))

                if len(self.data) >= 14:
                    action = (
                        model_1.predict(df.tail(6))[0]
                        + model_3.predict(df.tail(6))[0] / 2
                        # + model_3.predict(df.tail(6))[0] / 3
                    )
                    print("ENSEMBLE ACTION: {}".format(action))

                    if action[0] < 0:
                        # sell 10% of stock
                        percentage_to_sell = (
                            float(self.api.get_position(q.symbol).qty) * 0.1
                        )
                        print("selling stock")
                        self.api.submit_order(
                            symbol=q.symbol,
                            qty=percentage_to_sell,
                            side="sell",
                            type="market",
                            time_in_force="gtc",
                        )

                    if action[0] > 0:
                        # buy 10% of cash balance in stock
                        amount = (cash * 0.10) / q.close
                        if cash > amount * q.close:
                            print("buying stock")
                            self.api.submit_order(
                                symbol=q.symbol,
                                qty=amount,
                                side="buy",
                                type="market",
                                time_in_force="gtc",
                            )
                        else:
                            print("Not enough cash to buy stock")

            self.stream.subscribe_crypto_bars(print_crypto_trade, *tickers)

            self.stream.run()

        else:

            async def print_crypto_trade(q):
                account = self.api.get_account()
                cash = float(account.cash)
                print(account)

                self.close_values.append(q.close)
                self.volume_values.append(q.volume)
                self.data.append(
                    {
                        "open": q.open,
                        "close": q.close,
                        "rsi": TA.RSI(np.array(self.close_values), 14)[-1],
                        "sma": TA.SMA(np.array(self.close_values), 12)[-1],
                        "obv": TA.OBV(
                            np.array(self.close_values), np.array(self.volume_values)
                        )[-1],
                    }
                )

                df = pd.DataFrame(self.data)
                print("columns: {}".format(df.columns))
                df.fillna(0, inplace=True)
                print(self.close_values)
                print(len(self.close_values))

                # determine which model to use
                if "PPO" in model_names:
                    model = PPO.load(model_names[0])
                elif "TD3" in model_names:
                    model = TD3.load(model_names[0])
                else:
                    model = A2C.load(model_names[0])

                if len(self.data) >= 14:
                    action = model.predict(df.tail(6))

                    if action[0][0] < 0:
                        # sell 10% of stock
                        percentage_to_sell = (
                            float(self.api.get_position(q.symbol).qty) * 0.1
                        )
                        print("selling stock")
                        self.api.submit_order(
                            symbol=q.symbol,
                            qty=percentage_to_sell,
                            side="sell",
                            type="market",
                            time_in_force="gtc",
                        )

                    if action[0][0] > 0:
                        # buy 10% of cash balance in stock
                        amount = (cash * 0.10) / q.close
                        if cash > amount * q.close:
                            print("buying stock")
                            self.api.submit_order(
                                symbol=q.symbol,
                                qty=amount,
                                side="buy",
                                type="market",
                                time_in_force="gtc",
                            )
                        else:
                            print("Not enough cash to buy stock")

            self.stream.subscribe_crypto_bars(print_crypto_trade, *tickers)

            self.stream.run()


if __name__ == "__main__":
    # initiate the class for historical data
    # if you want to trade live you need to pass in the stream parameter
    trader = SmartTrader(
        api_key="PKP1ETYE7CJ159NMRJ9S",
        secret_key="teabIy9nt6drve1VXVmudbsb2TtLSSfNMkhZuIi9",
        endpoint="https://paper-api.alpaca.markets",
        api=tradeapi.REST(
            key_id="PKP1ETYE7CJ159NMRJ9S",
            secret_key="teabIy9nt6drve1VXVmudbsb2TtLSSfNMkhZuIi9",
            base_url=URL("https://paper-api.alpaca.markets"),
        ),
        stream=Stream(
            "PKP1ETYE7CJ159NMRJ9S",
            "teabIy9nt6drve1VXVmudbsb2TtLSSfNMkhZuIi9",
            base_url=URL("https://paper-api.alpaca.markets"),
            data_feed="iex",
        ),  # <- replace to 'sip' if you have PRO subscription
    )

    # get the data historical
    df = trader.get_data(
        symbol=[
            "NVDA",
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "NFLX",
            "NTDOY",
            "AMD",
            "SNOW",
            "MDB",
        ],
        timeframe=TimeFrame.Day,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1),
    )

    # train and run the model historical
    trader.train_and_run_model(
        df=df,
        model=PPO,
        total_timesteps=10,
        log_interval=10,
        ensemble=False,
        visualize=False,
    )

    # run the model live
    # model name should be in the format of trained_model_{model_name}_{day-month-last 2 digits of the current year}
    trader.run_model(model_names=["trained_model_PPO_31-10-22", "trained_model_A2C_30-10-22"], tickers=["BTCUSD"], ensemble=True)

