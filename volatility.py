#!/usr/bin/python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from river import forest, preprocessing, feature_extraction
from tqdm import trange

df = pd.read_csv("ty_ohlc.csv", index_col=0, parse_dates=True)
diffs = np.log(df["Close"]).diff().dropna()
squared_diff = diffs.pow(2).rename("squared_diff")
level = df["Close"].loc[diffs.index]
n_lags = 5
x = pd.concat(
    [squared_diff] 
    + [level.shift(1)] 
    + [diffs.shift(i).rename(str(i)) for i in range(1, n_lags + 1)],
    axis=1,
).fillna(0)
y = x.pop("squared_diff")

volatility_model = (
    feature_extraction.PolynomialExtender()
    | preprocessing.AdaptiveStandardScaler()
    | forest.AMFRegressor()
)

vol_ests = []
upper_band = []
lower_band = []
for i in trange(len(x)):
    date = x.index[i]
    x_i = x.iloc[i]
    y_i = y.iloc[i]
    volatility_model.learn_one(x_i, y_i)
    try:
        x_i_1 = x.iloc[i + 1]
        variance = volatility_model.predict_one(x_i_1)
        vol_est = np.sqrt(variance)
        vol_ests.append(vol_est)
        upper_band.append(level.iloc[i] * (1 +  2 * vol_est))
        lower_band.append(level.iloc[i] * (1 -  2 * vol_est))
    except IndexError: # last row
        pass

with open(f'volatility_model.pkl', 'wb') as f:
    pickle.dump(volatility_model, f)

with open(f'volatility_model.pkl', 'rb') as f:
    volatility_model = pickle.load(f)

upper_band = pd.Series(upper_band, index=level.index[1:])
lower_band = pd.Series(lower_band, index=level.index[1:])
vol_ests = pd.Series(vol_ests, index=level.index[1:])
vol_ests.to_csv('NVDA_vol_ests.csv')

plt.figure(figsize=(12, 6))
plt.plot(level.iloc[-100:], label="Level", c="k")
plt.plot(upper_band.iloc[-100:], label="Upper Band", c='r', linestyle='dashed')
plt.plot(lower_band.iloc[-100:], label="Lower Band", c='b', linestyle='dashed')
plt.title("Volatility Bands")
plt.legend()
plt.show()
plt.savefig("images/volatility.png")
