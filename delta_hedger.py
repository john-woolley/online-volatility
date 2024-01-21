# Description: Delta hedging environment for reinforcement learning
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces
from collections import deque
import pandas as pd
from sb3_contrib import RecurrentPPO as PPO
from stable_baselines3.common.monitor import Monitor

import pandas as pd
from scipy.stats import norm


class PutOption:
    def __init__(self, strike, time_to_expiry) -> None:
        self.strike = strike
        self.time_to_expiry = time_to_expiry

    def price(self, spot, volatility, rate):
        d1 = (
            np.log(spot / self.strike) + (0.5 * volatility**2) * self.time_to_expiry
        ) / (volatility * np.sqrt(self.time_to_expiry))
        d2 = d1 - volatility * np.sqrt(self.time_to_expiry)
        return self.strike * np.exp(-rate * self.time_to_expiry) * norm.cdf(
            -d2
        ) - spot * norm.cdf(-d1)

    def delta(self, spot, volatility):
        d1 = (
            np.log(spot / self.strike) + (0.5 * volatility**2) * self.time_to_expiry
        ) / (volatility * np.sqrt(self.time_to_expiry))
        return -norm.cdf(-d1)


class CallOption:
    def __init__(self, strike, time_to_expiry) -> None:
        self.strike = strike
        self.time_to_expiry = time_to_expiry

    def price(self, spot, volatility, rate):
        d1 = (
            np.log(spot / self.strike) + (0.5 * volatility**2) * self.time_to_expiry
        ) / (volatility * np.sqrt(self.time_to_expiry))
        d2 = d1 - volatility * np.sqrt(self.time_to_expiry)
        return spot * norm.cdf(d1) - self.strike * np.exp(
            -rate * self.time_to_expiry
        ) * norm.cdf(d2)

    def delta(self, spot, volatility):
        d1 = (
            np.log(spot / self.strike) + (0.5 * volatility**2) * self.time_to_expiry
        ) / (volatility * np.sqrt(self.time_to_expiry))
        return norm.cdf(d1)


class DeltaHedger(gym.Env):
    def __init__(
        self,
        data,
        initial_balance=100,
        expiry=252,
        n_lags=10,
        transaction_cost=0.0025,
        test=False
    ):
        super().__init__()
        self.data = data
        if test:
            self.periods = [self.data]
        else:
            self.periods = [
                self.data.iloc[i : i + expiry] for i, _ in enumerate(self.data.index)
            ]
        self.action_space = spaces.Box(low=-100, high=100, shape=(1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_lags, 8))
        self.initial_balance_multiplier = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.buffer_len = n_lags
        self.render_df = pd.DataFrame()
        self.expiry = expiry
        self.call_delta = 0
        self.put_delta = 0
        self.call_value = 0
        self.put_value = 0
        self.balance = 0

    @property
    def current_portfolio_value(self):
        return self.balance - self.call_value - self.put_value + self.hedge_value

    @property
    def options_delta(self):
        return -(self.call_delta + self.put_delta)

    @property
    def portfolio_delta(self):
        return self.options_delta + self.hedge_delta

    @property
    def hedge_value(self):
        return self.net_leverage * self.spot

    @property
    def hedge_delta(self):
        return self.net_leverage

    def reset(self, seed=42):
        self.var = 0
        self.net_leverage = 0

        self.current_step = 0
        if len(self.periods) > 1:
            self.period = random.choice(self.periods[:-1])
        else:
            self.period = self.periods[0]
        self.spot = self.period["Close"].iloc[0]
        self.rate = 0.05
        self.strike = self.spot * np.exp(self.rate)
        self.time_to_expiry = len(self.period)
        self.volatility = self.period["volatility"].iloc[0]

        self.put_option = PutOption(self.strike, 1)
        self.call_option = CallOption(self.strike, 1)
        self.straddle_size = 100
        self.call_value = (
            self.call_option.price(self.spot, self.volatility, self.rate)
            * self.straddle_size
        )
        self.put_value = (
            self.put_option.price(self.spot, self.volatility, self.rate)
            * self.straddle_size
        )
        self.initial_balance = (
            self.put_value
            + self.call_value
            + self.initial_balance_multiplier * self.spot
        )
        self.balance = self.initial_balance
        self.put_delta = self.put_option.delta(self.spot, self.volatility)
        self.call_delta = self.call_option.delta(self.spot, self.volatility)
        self.log_ret = 0
        self.state_buffer = deque([], self.buffer_len)
        self.paid_slippage = 0
        state_frame = self._get_state_frame()
        while len(self.state_buffer) < self.buffer_len:
            self.state_buffer.append(state_frame)
        return self._get_observation(), {}

    def _get_state_frame(self) -> np.ndarray:
        return np.array(
            [
                self.spot,
                self.strike,
                self.portfolio_delta,
                self.options_delta,
                self.hedge_delta,
                self.call_value,
                self.put_value,
                self.hedge_value,
            ]
        )

    def _trade(self, amount: float, sign: int) -> None:
        """
        Executes a trade of the given amount and sign.
        Sign should be 1 for buying and -1 for selling.
        """
        targeted_delta = self.portfolio_delta + amount * sign
        targeted_cost = (
            amount * sign * self.spot * np.sign(self.net_leverage)
            + amount * self.spot * self.transaction_cost
        )
        if (
            (abs(targeted_delta) < 5 or abs(targeted_delta) < abs(self.portfolio_delta))
            and (self.current_portfolio_value > targeted_cost)
            and self.balance > targeted_cost
        ):
            self.paid_slippage = amount * self.spot * self.transaction_cost
            self.net_leverage += amount * sign
            self.balance -= amount * sign * self.spot + self.paid_slippage

    def _update_options(self) -> None:
        self.prev_put_value = self.put_value
        self.prev_call_value = self.call_value
        self.put_option.time_to_expiry -= 1 / self.expiry
        self.call_option.time_to_expiry -= 1 / self.expiry
        self.call_value = (
            self.call_option.price(self.spot, self.volatility, self.rate)
            * self.straddle_size
        )
        self.put_value = (
            self.put_option.price(self.spot, self.volatility, self.rate)
            * self.straddle_size
        )
        self.call_delta = (
            self.call_option.delta(self.spot, self.volatility) * self.straddle_size
        )
        self.put_delta = (
            self.put_option.delta(self.spot, self.volatility) * self.straddle_size
        )
        option_pnl = (self.prev_put_value - self.put_value) + (
            self.prev_call_value - self.call_value
        )
        self.option_pnl = option_pnl

    def _get_reward(self) -> float:
        return (
            np.log(self.current_portfolio_value / self.initial_balance)
            if self.current_portfolio_value > 0
            else -1
        )

    def _accrue_interest(self) -> None:
        self.balance *= 1 + (self.rate / self.expiry)

    def step(self, action) -> tuple:
        self.prev_portfolio_value = self.current_portfolio_value
        self.prev_spot = self.spot
        self.spot = self.period["Close"].iloc[self.current_step]
        self.volatility = self.period["volatility"].iloc[self.current_step]
        sign = 1 if action[0] > 0 else -1
        amount = np.abs(action[0])
        self._accrue_interest()
        self._trade(amount, sign)
        self._update_options()
        reward = self._get_reward()
        self.var = (
            self.volatility / np.sqrt(252) * self.spot * self.portfolio_delta
        ) / self.current_portfolio_value
        self.render(action, mode=None)
        self.state_buffer.append(self._get_state_frame())
        self.current_step += 1
        done = (
            self.current_step == len(self.period) - 1
            or self.current_portfolio_value < 0
        )
        self.done = done
        if done:
            self.reset()

        info = {}
        obs = self._get_observation()
        return obs, reward, done, False, info

    def _get_observation(self):
        return self.state_buffer

    def render(self, action, mode="human"):
        current_date = self.period.index[self.current_step]
        today_action = "buy" if action[0] > 0 else "sell"
        today_size = action[0]
        current_price = self.period.iloc[self.current_step]["Close"]

        if mode == "human":
            print(
                (
                    f"Step:{self.current_step}, \
                  Date: {current_date}, \
                  Market Value: {self.current_portfolio_value:.2f}, \
                  Balance: {self.balance:.2f}, \
                  Stock Owned: {self.net_leverage}, \
                  Stock Price: {current_price:.2f}, \
                  Today Action: {today_action}:{today_size}. \
                  paid_slippage: {self.paid_slippage}, \
                  volatility: {self.volatility}, \
                  put delta: {self.put_delta}, \
                    call delta: {self.call_delta}, \
                  options delta: {self.options_delta}, \
                portfolio delta: {self.portfolio_delta}"
                )
            )
        else:
            pass
        dict = {
            "Date": [current_date],
            "market_value": [self.current_portfolio_value],
            "balance": [self.balance],
            "stock_owned": [self.net_leverage],
            "hedge_position": [self.net_leverage * self.spot],
            "price": [current_price],
            "action": [today_action],
            "amount": [today_size],
            "paid_slippage": [self.paid_slippage],
            "vol_est": [self.volatility],
            "options_delta": [self.options_delta],
            "portfolio_delta": [self.portfolio_delta],
            "put_value": [self.put_value],
            "call_value": [self.call_value],
        }
        step_df = pd.DataFrame.from_dict(dict)
        self.render_df = pd.concat([self.render_df, step_df], ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("NVDA_ohlcv.csv", index_col=0, parse_dates=True)
    train = df.iloc[:-252]
    test = df.iloc[-252:]
    mfile = "delta_hedger"
    env = DeltaHedger(train)
    log_dir = "log/"
    env = Monitor(env, log_dir, override_existing=False)
    # model_train = PPO("MlpLstmPolicy", env, verbose=1)
    model_train = PPO.load("ppo_delta_hedger", env=env)
    model_train.learn(total_timesteps=10000, progress_bar=True)
    model_train.save("ppo_delta_hedger")
    env_test = DeltaHedger(test, expiry=252, test=True)
    model_test = PPO.load("ppo_delta_hedger", env=env_test)
    vec_env = model_test.get_env()
    obs = vec_env.reset()
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    for i in range(len(test["Close"]) - 1):
        action, lstm_states = model_test.predict(
            obs, state=lstm_states, episode_start=episode_starts
        )
        obs, reward, done, info = vec_env.step(action)
        if done:
            break
    env_test.render_df.to_csv("ppo_delta_hedger.csv")
