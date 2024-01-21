import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy


def moving_average(values: np.array, window: int):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


x, y = ts2xy(load_results('log/'), "timesteps")
print(x, y) 
y = moving_average(y, window=100)
x = x[len(x) - len(y):]
fig = plt.figure('Learning Curve Smoothed')
plt.plot(x, y)
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
fig.savefig('learning_curve2.png')

render_df = pd.read_csv("ppo_delta_hedger.csv")
df = render_df.set_index("Date")
fig, ax = plt.subplots(figsize=(18, 6))
df.plot(y="market_value", use_index=True, ax=ax, color="lightgrey")
df.plot(y="portfolio_delta", use_index=True, ax=ax, secondary_y=True, style='--', color="purple")
df.plot(y="options_delta", use_index=True, ax=ax, secondary_y=True, style = '--', color="red")


plt.savefig(f"test_render2.png")
