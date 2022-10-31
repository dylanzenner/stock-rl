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
    """A stock trading environment for interacting with the historical and live alpaca markets API"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.reward_range = (-sys.maxsize, sys.maxsize)
        # we start with a 25,000 balance to avoid pattern day trading
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.net_worth = 0
        self.profit = 0
        self.df = df
        self.graph_balance = []
        self.graph_reward = []
        self.previous_net_worth = 0

        # Actions for BUY, SELL, HOLD
        # using Box here because all 3 algorithms are able to work with the box action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype="float32")

        # the observation will be the last 1 data points
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 5), dtype="float32"
        )

    def _next_observation(self):
        # need to figure out how to wait for the data to be populated so we can return the dataframe for the next observation

        frame = np.array(
            [
                self.df.loc[self.current_step : self.current_step + 4, "open"],
                self.df.loc[self.current_step : self.current_step + 4, "close"],
                self.df.loc[self.current_step : self.current_step + 4, "rsi"],
                self.df.loc[self.current_step : self.current_step + 4, "sma"],
                self.df.loc[self.current_step : self.current_step + 4, "obv"],
            ]
        )
        # print('-----------')
        # print(frame)
        # print('-----------')

        # Append account info
        obs = np.append(
            frame,
            [
                [
                    self.balance,
                    self.stock_held,
                    self.trades_made,
                    self.net_worth,  # place holder for now
                    self.profit,  # place holder for now
                ]
            ],
            axis=0,
        )
        # print('-----------')
        # print(obs)
        # print('-----------')
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
                self.graph_balance.append(self.balance)
                self.stock_held += stock_purchased
                self.trades_made += stock_purchased
            else:
                pass

        if action[0] < 0 and self.df.loc[self.current_step, "rsi"] > 70:
            # Sell 10% of stock held
            stocks_sold = int(self.stock_held * 0.10)

            self.trades_made += stocks_sold
            self.balance += int(stocks_sold * current_price)
            self.graph_balance.append(self.balance)
            self.stock_held -= stocks_sold

        # self.net_worth = self.balance + self.stock_held_held * current_price

    def step(self, action):

        # Execute one time step within the environment
        self._take_action(action)
        self.graph_reward.append(self.current_step)
        self.current_step += 1

        # # if the current step ever goes above the length of the dataframe, then reset to 0
        if self.current_step > len(self.df.loc[:, "open"].values) - 6:
            self.current_step = 0

        reward = self.balance

        done = self.balance <= 0

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.data = []

        # set the current step to the first point in the dataframe
        self.current_step = random.randint(0, len(self.df.loc[:, "open"].values) - 6)
        return self._next_observation()

    def render(self, mode="human", close=False):
        print("-------------------------------")
        print("Balance: {}".format(self.balance))
        print("Stocks held: {}".format(self.stock_held))
        print("Trades made: {}".format(self.trades_made))
