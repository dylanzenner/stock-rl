import random
import json
import gym
from gym import spaces
import pandas as pd
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import talib as TA


class StockTradingEnv(gym.Env):
    """A stock trading environment for training an agent to predict when to buy and sell stocks utilizing the Alpaca Markets API"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.reward_range = (-sys.maxsize, sys.maxsize)
        # start with a 25,000 balance to avoid pattern day trading if trading with real money
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.net_worth = 10000
        self.profit = 0
        self.df = df

        # Actions for BUY, SELL, HOLD
        # using Box here because all 3 algorithms are able to work with the box action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype="float32")

        # the observation will be the last 6 data points
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6, 5), dtype="float32"
        )

    def _next_observation(self):

        frame = np.array(
            [
                self.df.loc[self.current_step : self.current_step + 4, "open"],
                self.df.loc[self.current_step : self.current_step + 4, "close"],
                self.df.loc[self.current_step : self.current_step + 4, "rsi"],
                self.df.loc[self.current_step : self.current_step + 4, "sma"],
                self.df.loc[self.current_step : self.current_step + 4, "obv"],
            ]
        )


        # Append account info
        obs = np.append(
            frame,
            [
                [
                    self.balance,
                    self.stock_held,
                    self.trades_made,
                    self.net_worth,
                    self.profit,
                ]
            ],
            axis=0,
        )

        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"],
            self.df.loc[self.current_step, "close"],
        )

        if action[0] > 0 and self.df.loc[self.current_step, "rsi"] < 30:

            # Buy 10% of balance in stock as long as we are spending at least $1
            # This is due to alpaca markets requiring a minimum of $1 to be spent per trade
            stock_purchased = self.balance * 0.10 / current_price
            if stock_purchased >= 1:
                self.balance -= stock_purchased * current_price
                self.stock_held += stock_purchased
                self.trades_made += stock_purchased

        elif action[0] < 0 and self.df.loc[self.current_step, "rsi"] > 70:

            # Sell 10% of stock held
            stocks_sold = int(self.stock_held * 0.10)

            self.trades_made += stocks_sold
            self.balance += int(stocks_sold * current_price)
            self.stock_held -= stocks_sold

        self.net_worth = self.balance + self.stock_held * current_price
        self.profit = self.net_worth - self.balance

    def step(self, action):

        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        # # if the current step ever goes above the length of the dataframe, then reset to 0
        if self.current_step > len(self.df.loc[:, "open"].values) - 6:
            self.current_step = 0

        reward = self.profit

        if self.net_worth <= 0:
            done = True
        else:
            done = False

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.net_worth = 10000
        self.profit = 0

        # set the current step to the first point in the dataframe
        self.current_step = random.randint(0, len(self.df.loc[:, "open"].values) - 6)
        return self._next_observation()

    def render(self, mode="human", close=False):
        # print("-------------------------------")
        balance = self.balance
        stock_held = self.stock_held
        trades_made = self.trades_made
        net_worth = self.net_worth
        profit = self.profit
        cumulative_return = (self.net_worth - 10000) / 10000
        annualized_return = (self.net_worth - 10000 / 10000) ** (1 / (self.df.shape[0] / 252)) - 1
        annualized_volatility = self.df.loc[:, "percent_change"].std() * np.sqrt(252)
        sharpe_ratio = (cumulative_return - (cumulative_return * 0.03)) / (self.df.loc[:, "percent_change"].std() * np.sqrt(252))
        
        print("Balance: {}".format(balance))
        print("Stocks held: {}".format(stock_held))
        print("Trades made: {}".format(trades_made))
        print("Net Worth: {}".format(net_worth))
        print("Profit: {}".format(profit))
        print("Cumulative Return: {}".format(cumulative_return))
        print("Annualized Return: {}".format(annualized_return))
        print("Annualized Volatility: {}".format(annualized_volatility))
        print('Sharpe Ratio: {}'.format(sharpe_ratio))
        Roll_Max = self.df['close'].cummax()
        Daily_Drawdown = self.df['close']/ Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        print('Max Daily Drawdown: {}'.format(Max_Daily_Drawdown.min()))
