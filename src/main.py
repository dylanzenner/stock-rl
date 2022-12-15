from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, TD3, A2C, DDPG
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
import boto3
import os

log_dir = "logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


secret_client = boto3.client("secretsmanager")


def retrieve_secret(secret_name, id=None, secret=None):
    get_secret_value_response = secret_client.get_secret_value(
        SecretId=secret_name)
    if id:
        api_key = eval(get_secret_value_response["SecretString"])
        return api_key["api_key"]
    else:
        api_secret = eval(get_secret_value_response["SecretString"])
        return api_secret["api_secret"]


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
            symbol_or_symbols=[
                *symbol] if type(symbol) == type([]) else [symbol],
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
        df['percent_change'] = df['close'].pct_change()
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

    def train_model(
        self, df, model, total_timesteps, log_interval, ensemble, visualize=False
    ):
        if ensemble:
            # train the models
            for i in model:
                if i == PPO:
                    name_var = "PPO"
                elif i == TD3:
                    name_var = "TD3"
                elif i == DDPG:
                    name_var = 'DDPG'
                else:
                    name_var = "A2C"
                env = DummyVecEnv([lambda: StockTradingEnv(df)])

                n_actions = env.action_space.shape[-1]
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )

                model_name = (
                    i("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
                    if i in (PPO, A2C)
                    else i(
                        "MlpPolicy",
                        env,
                        verbose=1,
                        action_noise=action_noise,
                        tensorboard_log="logs/",
                        train_freq=(1, "step"),
                    )
                )

                model_name.learn(
                    total_timesteps=total_timesteps,
                    log_interval=log_interval,
                    reset_num_timesteps=False,
                    tb_log_name=name_var,
                )
                model_name.save(
                    "trained_model_{}_{}".format(
                        name_var,
                        datetime.now().strftime("%d-%m-%y"),
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
                model_3 = model[2].load(self.models_to_ensemble[2])
                print(model_1, model_2, model_3)

                obs = env.reset()
                for i in range(10000):
                    # print("---------------------------")
                    action = (
                        model_1.predict(obs)[0]
                        + model_2.predict(obs)[0]
                        + model_3.predict(obs)[0]
                    ) / 3
                    obs, rewards, done, info = env.step(action)
                env.render()

        else:
            if type(model) != type([]) and model in (PPO, A2C):
                if model == PPO:
                    name_var = "PPO"
                else:
                    name_var = "A2C"
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                model = model("MlpPolicy", env, verbose=1,
                              tensorboard_log="logs/")

                model.learn(
                    total_timesteps=total_timesteps,
                    log_interval=log_interval,
                    tb_log_name=name_var,
                    reset_num_timesteps=False,
                )
                model.save(
                    "trained_model_{}_{}".format(
                        name_var, datetime.now().strftime("%d-%m-%y")
                    )
                )

                if visualize:
                    obs = env.reset()
                    for i in range(10000):
                        # print("---------------------------")
                        action, _states = model.predict(obs)
                        obs, rewards, done, info = env.step(action)
                    env.render()

            elif model in (DDPG, TD3):
                env = DummyVecEnv([lambda: StockTradingEnv(df)])
                n_actions = env.action_space.shape[-1]
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
                if model == TD3:
                    name = "TD3"
                else:
                    name = "DDPG"
                model = model(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    action_noise=action_noise,
                    tensorboard_log="logs/",
                    train_freq=(1, "step"),
                )

                model.learn(
                    total_timesteps=total_timesteps,
                    log_interval=log_interval,
                    reset_num_timesteps=True,
                    tb_log_name=name,
                )
                model.save(
                    "trained_model_{}_{}".format(
                        name, datetime.now().strftime("%d-%m-%y")
                    )
                )

                if visualize:
                    obs = env.reset()
                    for i in range(10000):
                        # print("---------------------------")
                        action, _states = model.predict(obs)
                        obs, rewards, done, info = env.step(action)
                    env.render()

    def run_model(self, model_names, ensemble=None, tickers=None):
        model_1, model_2, model_3 = None, None, None
        if ensemble is True:
            for model in model_names:
                if "PPO" in model:
                    model_1 = PPO.load(model)
                elif "A2C" in model:
                    model_2 = A2C.load(model)
                else:
                    try:
                        model_3 = TD3.load(model)
                    except Exception as e:
                        model_3 = DDPG.load(model)

            async def make_trade(q):
                account = self.api.get_account()
                cash = float(account.cash)
                print(account)

                self.close_values.append(q.close)
                self.volume_values.append(q.volume)
                self.data.append(
                    {
                        "open": q.open,
                        "close": q.close,
                        "rsi": TA.RSI(np.array(self.close_values, dtype=float), 14)[-1],
                        "sma": TA.SMA(np.array(self.close_values, dtype=float), 12)[-1],
                        "obv": TA.OBV(
                            np.array(self.close_values, dtype=float),
                            np.array(self.volume_values, dtype=float),
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
                        + model_2.predict(df.tail(6))[0]
                        + model_3.predict(df.tail(6))[0]
                    ) / 3
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
                            time_in_force="day",
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
                                time_in_force="day",
                            )
                        else:
                            print("Not enough cash to buy stock")

            self.stream.subscribe_bars(make_trade, *tickers)

            self.stream.run()

        else:

            async def make_trade(q):
                account = self.api.get_account()
                cash = float(account.cash)
                print(account)

                self.close_values.append(q.close)
                self.volume_values.append(q.volume)
                self.data.append(
                    {
                        "open": q.open,
                        "close": q.close,
                        "rsi": TA.RSI(np.array(self.close_values, dtype=float), 14)[-1],
                        "sma": TA.SMA(np.array(self.close_values, dtype=float), 12)[-1],
                        "obv": TA.OBV(
                            np.array(self.close_values, dtype=float),
                            np.array(self.volume_values, dtype=float),
                        )[-1],
                    }
                )

                df = pd.DataFrame(self.data)
                print("columns: {}".format(df.columns))
                df.fillna(0, inplace=True)
                print(self.close_values)
                print(len(self.close_values))

                # determine which model to use
                if "PPO" in model_names[0]:
                    model = PPO.load(model_names[0])
                elif "TD3" in model_names[0]:
                    model = TD3.load(model_names[0])
                elif 'DDPG' in model_names[0]:
                    model = DDPG.load(model_names[0])
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
                            time_in_force="day",
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
                                time_in_force="day",
                            )
                        else:
                            print("Not enough cash to buy stock")

            self.stream.subscribe_bars(make_trade, *tickers)
            self.stream.run()


if __name__ == "__main__":

    # alpaca_key = retrieve_secret("alpaca_keys", id="api_key")
    # alpaca_secret = retrieve_secret("alpaca_keys", secret="api_secret")

    # initiate the class for historical data
    # if you want to trade live you need to pass in the stream parameter
    trader = SmartTrader(
        api_key='PKIP99J0RO3VLAS1GLBV',
        secret_key='Yhq6IrRceM2XHAGCW8DOs4kUUC2nrhZqbVhVOMHS',
        endpoint="https://paper-api.alpaca.markets",
        api=tradeapi.REST(
            key_id='PKIP99J0RO3VLAS1GLBV',
            secret_key='Yhq6IrRceM2XHAGCW8DOs4kUUC2nrhZqbVhVOMHS',
            base_url=URL("https://paper-api.alpaca.markets"),
        ),
        stream=Stream(
            'PKIP99J0RO3VLAS1GLBV',
            'Yhq6IrRceM2XHAGCW8DOs4kUUC2nrhZqbVhVOMHS',
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
            "SFIX",
            "CHWY",
            "ABNB",
            "TSLA",
            "DIS",
            "SHOP",
            "SONY",
            "CRM",
            "ROKU",
            "DDOG",
            "PINS",
            "DBX",
            "ATVI",
            "META",
            "ACN",
            "CSCO",
            "ADBE",
            "IBM",
            "SAP",
            "INTC",
            "ORCL",
            "MDB",
        ],
        timeframe=TimeFrame.Day,
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2022, 1, 1),
    )

    # train and run the model historical
    trader.train_model(
        df=df,
        model=DDPG,  # use a list if you wish to train multiple models, otherwise just one
        total_timesteps=10000,
        log_interval=1,
        ensemble=False,
        visualize=True,
    )

    # run the model live
    # model name should be in the format of trained_model_{model_name}_{day-month-last 2 digits of the current year}
    trader.run_model(
        model_names=["trained_model_DDPG_13-12-22"], # a list containing the algorihtm / algorithms to use
        tickers=[
            "NVDA",
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "NFLX",
            "NTDOY",
            "AMD",
            "SNOW",
            "SFIX",
            "CHWY",
            "ABNB",
            "TSLA",
            "DIS",
            "SHOP",
            "SONY",
            "CRM",
            "ROKU",
            "DDOG",
            "PINS",
            "DBX",
            "ATVI",
            "META",
            "ACN",
            "CSCO",
            "ADBE",
            "IBM",
            "SAP",
            "INTC",
            "ORCL",
            "MDB",
        ],
        ensemble=False, # either None or a bool value
    )