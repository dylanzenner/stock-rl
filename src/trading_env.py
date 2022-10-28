import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        # self.seed()
        self.df = df
        self.reward_range = (-sys.maxsize, sys.maxsize)
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.net_worth = 0

        # render profit vs reward
        self.graph_reward = []
        self.graph_profit = []
        self.graph_steps = []


        # Actions for BUY, SELL, HOLD
        # using Box here because all 3 algorithms are able to work with the box action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1, ),dtype='float32')

        # the observation will be the last 5 data points
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 5),dtype='float32')


    # Something is wrong here I keep getting the error:
    # ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 2 and the array at index 1 has size 5
    def _next_observation(self):
        # observation will be the last 5 data points
        frame = np.array([
            self.df.loc[self.current_step - 4:self.current_step, 'open'],
            self.df.loc[self.current_step - 4:self.current_step, 'close'],
            self.df.loc[self.current_step - 4:self.current_step, 'rsi'],
            self.df.loc[self.current_step - 4:self.current_step, 'sma'],
            self.df.loc[self.current_step - 4:self.current_step, 'obv'],

        ])
        print('--------------------')
        print(frame)
        print('--------------------')



        # # Append account info
        obs = np.append(frame, [[self.balance,
                                 self.stock_held ,
                                 self.trades_made,
                                 self.net_worth, # place holder for now
                                 self.balance, # place holder for now
                                 ]],
                                 axis=0
                                )
        print('--------------------')
        print(obs)
        print('--------------------')

        return obs

        

    def _take_action(self, action):
        # either buy or sell a stock
        current_price = random.uniform(
            self.df.loc[self.current_step, 'open'],
            self.df.loc[self.current_step, 'close'])

        if action[0] > 0:
            # Buy 10% of balance in stock
            stock_purchased = self.balance * .10 / current_price
            
        
            if self.balance > stock_purchased * current_price:
                self.balance -= stock_purchased * current_price
                self.stock_held += stock_purchased
                self.trades_made += stock_purchased


        if action[0] < 0:
            # Sell 10% of stock held
            stocks_sold = self.stock_held * .10
            
            self.trades_made += stocks_sold
            self.balance += stocks_sold * current_price
            self.stock_held -= stocks_sold

        # self.net_worth = self.balance + self.stock_held_held * current_price

        

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.graph_steps.append(self.current_step)
        self.current_step += 1
        

        # if the current step ever goes above the length of the dataframe, then reset to 0
        if self.current_step > len(self.df.loc[:, 'open'].values) - 5:
            self.current_step = 0
        
        reward = self.balance
        print('reward: {}'.format(reward))
        self.graph_reward.append(reward)
        print('graph_reward: {}'.format(self.graph_reward))
        done = self.net_worth <= 0

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = 10000
        self.trades_made = 0
        self.stock_held = 0
        self.graph_reward = []
        self.graph_profit = []
        self.graph_steps = []

        # set the current step to the first point in the dataframe
        self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - 5)



        return self._next_observation()
        

        

    def render(self, mode='human', close=False):
        print('-------------------------------')
        print('Balance: {}'.format(self.balance))
        print('Stocks held: {}'.format(self.stock_held))
        print('Trades made: {}'.format(self.trades_made))
        print('Reward: {}'.format(self.graph_reward))
        


        