import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib as mpt
from scipy.stats import norm

# Set the font dictionaries (for plot title and axis titles)
title_font = {
    "fontname": "Arial",
    "size": "16",
    "color": "black",
    "weight": "normal",
    "verticalalignment": "bottom",
}  # Bottom vertical alignment for more space
axis_font = {"fontname": "Arial", "size": "14"}

# Set the font properties (for use in legend)
font_path = "C:\Windows\Fonts\Arial.ttf"
font_prop = mpt.font_manager.FontProperties(fname=font_path, size=14)


class GarmanKohlhagenOption:
    def __init__(
        self, strike, spot, volatility, time_to_expiry, domestic_rate, foreign_rate
    ) -> None:
        self.strike = strike
        self.spot = spot
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate
        self.volatility = volatility
        self.tau = time_to_expiry

    @property
    def d1(self):
        return (
            np.log(self.spot / self.strike)
            + (self.domestic_rate - self.foreign_rate + 0.5 * self.volatility**2)
            * self.tau
        ) / (self.volatility * np.sqrt(self.tau))

    @property
    def d2(self):
        return self.d1 - self.volatility * np.sqrt(self.tau)

    def update(self, spot, volatility, time_to_expiry, domestic_rate, foreign_rate):
        self.spot = spot
        self.volatility = volatility
        self.tau = time_to_expiry
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate


class GarmanKohlhagenCall(GarmanKohlhagenOption):
    def __init__(
        self, strike, spot, volatility, time_to_expiry, domestic_rate, foreign_rate
    ) -> None:
        self.strike = strike
        self.spot = spot
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate
        self.volatility = volatility
        self.tau = time_to_expiry

    @property
    def price(self):
        return np.exp(-self.foreign_rate * self.tau) * self.spot * norm.cdf(
            self.d1
        ) - np.exp(-self.domestic_rate * self.tau) * self.strike * norm.cdf(self.d2)

    @property
    def delta(self):
        return norm.cdf(self.d1)


class GarmanKohlhagenPut(GarmanKohlhagenOption):
    def __init__(
        self, strike, spot, volatility, time_to_expiry, domestic_rate, foreign_rate
    ) -> None:
        self.strike = strike
        self.spot = spot
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate
        self.volatility = volatility
        self.tau = time_to_expiry

    @property
    def price(self):
        return np.exp(-self.domestic_rate * self.tau) * self.strike * norm.cdf(
            -self.d2
        ) - np.exp(-self.foreign_rate * self.tau) * self.spot * norm.cdf(-self.d1)

    @property
    def delta(self):
        return -np.exp(-self.foreign_rate * self.tau) * norm.cdf(-self.d1)


class Curve:
    def __init__(self, time: np.ndarray, rate: np.ndarray):
        self.time = time
        self.rate = rate

    def __sub__(self, other):
        return Curve(self.time, self.rate - other.rate)

    def __add__(self, other):
        return Curve(self.time, self.rate + other.rate)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(df["time"].values, df["rate"].values)

    @classmethod
    def from_csv(cls, csv_file: str):
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)

    def interpolate(self, time: float):
        return np.interp(time, self.time, self.rate)

    def discount_factor(self, time: float):
        return np.exp(-self.interpolate(time) * time)

    @property
    def discount_factors(self):
        return np.exp(-self.rate * self.time)
    
    def loc(self, time: float):
        return self.rate[np.where(self.time == time)][0]


class VolCurve:
    def __init__(self, time: np.ndarray, vol: np.ndarray):
        self.time = time
        self.vol = vol

    def interpolate(self, time: float):
        return np.interp(time, self.time, self.vol)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(df["time"].values, df["vol"].values)

    @classmethod
    def from_csv(cls, csv_file: str):
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)
    
    def loc(self, time: float):
        return self.vol[np.where(self.time == time)][0]


class VannaVolga:
    def __init__(
        self,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        atm_vols: VolCurve,
        put_vols: VolCurve,
        call_vols: VolCurve,
        delta: float = 0.25,
    ):
        self.delta = delta
        self.spot = spot
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.atm_vols = atm_vols
        self.put_vols = put_vols
        self.call_vols = call_vols
        self.df_diffs = self._get_df_diffs()
        self.forwards = self._get_forward_rates()

        self.atm_strikes = Curve(domestic_curve.time, self._get_atm_strikes())
        self.put_strikes = Curve(domestic_curve.time, self._get_put_strikes())
        self.call_strikes = Curve(domestic_curve.time, self._get_call_strikes())

        self.strikes = (1 + np.arange(-0.15, 0.16, 0.01)) * self.spot

        self.vol_surface = self._build_vol_surface()
        

    def _get_alpha(self):
        return -st.norm.ppf(self.delta * (1 / self.foreign_curve.discount_factors))

    def _get_forward_rates(self):
        return self.spot * np.exp(np.multiply(self.df_diffs, self.domestic_curve.time))

    def _get_df_diffs(self):
        return (
            self.domestic_curve.discount_factors - self.foreign_curve.discount_factors
        )

    def _get_atm_strikes(self):
        return self.forwards * np.exp(0.5 * self.atm_vols.vol**2 * self.atm_vols.time)

    def _get_put_strikes(self):
        return self.forwards * np.exp(
            -self._get_alpha() * self.put_vols.vol * np.sqrt(self.put_vols.time)
            + 0.5 * self.put_vols.vol**2 * self.put_vols.time
        )
    
    def _get_call_strikes(self):
        return self.forwards * np.exp(
            self._get_alpha() * self.call_vols.vol * np.sqrt(self.call_vols.time)
            + 0.5 * self.call_vols.vol**2 * self.call_vols.time
        )

    def _get_d1(self, strike: float, time: float):
        return (np.log(self.spot / strike) + 0.5 * self.atm_vols.loc(time) **2 * time) / (
            self.atm_vols.loc(time) * np.sqrt(time)
        )

    def _get_d2(self, strike: float, time: float):
        return self._get_d1(strike, time) - self.atm_vols.loc(time) * np.sqrt(time)

    def _get_z1(self, strike, time):
        return (
            np.log(self.atm_strikes.loc(time) / strike)
            * np.log(self.call_strikes.loc(time) / strike)
            / (
                np.log(self.atm_strikes.loc(time) / self.put_strikes.loc(time))
                * np.log(self.call_strikes.loc(time) / self.put_strikes.loc(time))
            )
        )

    def _get_z2(self, strike, time):
        return (
            np.log(strike / self.put_strikes.loc(time))
            * np.log(self.call_strikes.loc(time) / strike)
            / (
                np.log(self.atm_strikes.loc(time) / self.put_strikes.loc(time))
                * np.log(self.call_strikes.loc(time) / self.atm_strikes.loc(time))
            )
        )

    def _get_z3(self, strike, time):
        return (
            np.log(strike / self.put_strikes.loc(time))
            * np.log(strike / self.atm_strikes.loc(time))
            / (
                np.log(self.call_strikes.loc(time) / self.put_strikes.loc(time))
                * np.log(self.call_strikes.loc(time) / self.atm_strikes.loc(time))
            )
        )

    def _get_first_order_approx(self, strike, time):
        return (
            self._get_z1(strike, time) * self.put_vols.loc(time)
            + self._get_z2(strike, time) * self.atm_vols.loc(time)
            + self._get_z3(strike, time) * self.call_vols.loc(time)
            - self.atm_vols.loc(time)
        )

    def _get_second_order_approx(self, strike, time):
        return (
            self._get_z1(strike, time)
            * self._get_d1(self.put_strikes.loc(time), time)
            * self._get_d2(self.put_strikes.loc(time), time)
            * (self.put_vols.loc(time) - self.atm_vols.loc(time)) ** 2
            + self._get_z2(strike, time)
            * self._get_d1(self.atm_strikes.loc(time), time)
            * self._get_d2(self.atm_strikes.loc(time), time)
            * (self.atm_vols.loc(time) - self.atm_vols.loc(time)) ** 2
            + self._get_z3(strike, time)
            * self._get_d1(self.call_strikes.loc(time), time)
            * self._get_d2(self.call_strikes.loc(time), time)
            * (self.call_vols.loc(time) - self.atm_vols.loc(time)) ** 2
        )

    def _get_d1_d2(self, strike, time):
        return self._get_d1(strike, time) * self._get_d2(strike, time)

    def _get_vol(self, strike, time):
        return self.atm_vols.loc(time) + (
            -self.atm_vols.loc(time)
            + np.sqrt(
                self.atm_vols.loc(time)**2
                + self._get_d1_d2(strike, time)
                * (
                    2 * self.atm_vols.loc(time) * self._get_first_order_approx(strike, time)
                    + self._get_second_order_approx(strike, time)
                )
            )
        ) / (self._get_d1_d2(strike, time))
    
    def _build_vol_surface(self):
        vol_surface = np.zeros((31, 16), dtype=float)
        for i in range(31):
            for j in range(16):
                vol_surface[i][j] = self._get_vol(self.strikes[i], self.domestic_curve.time[j])
        return vol_surface


if __name__ == "__main__":

    spot = 1.5
    # Time tenor points
    tenors = np.array(
        [
            0.0192,
            0.0384,
            0.0877,
            0.1726,
            0.2493,
            0.3333,
            0.4166,
            0.5,
            0.75,
            1,
            1.25,
            1.5,
            2,
            3,
            4,
            5,
        ]
    )

    # Hypothetical volatility quotes for three most liquid points on the vol surface 25D Put, 25D Call and ATM
    vol_25d_put = np.array(
        [
            0.121,
            0.1215,
            0.1105,
            0.113,
            0.1224,
            0.1236,
            0.125,
            0.116,
            0.1175,
            0.1322,
            0.136,
            0.14,
            0.1411,
            0.1433,
            0.1445,
            0.145,
        ]
    )
    vol_25d_call = np.array(
        [
            0.121,
            0.1215,
            0.1105,
            0.113,
            0.1224,
            0.1236,
            0.125,
            0.116,
            0.1175,
            0.1322,
            0.136,
            0.14,
            0.1411,
            0.1433,
            0.1445,
            0.145,
        ]
    )
    vol_atm = np.array(
        [
            0.118,
            0.1182,
            0.1015,
            0.1029,
            0.115,
            0.116,
            0.118,
            0.105,
            0.108,
            0.121,
            0.124,
            0.132,
            0.135,
            0.1375,
            0.14,
            0.141,
        ]
    )

    # Hypothetical interest rates in two currencies
    domestic_rates = np.array(
        [
            0.005,
            0.0052,
            0.0059,
            0.006,
            0.0063,
            0.0069,
            0.007,
            0.0072,
            0.0075,
            0.0077,
            0.008,
            0.0085,
            0.009,
            0.00925,
            0.0095,
            0.0098,
        ]
    )

    foreign_rates = np.array(
        [
            0.0043,
            0.004,
            0.005,
            0.0055,
            0.0068,
            0.0071,
            0.0066,
            0.0078,
            0.0085,
            0.0083,
            0.0088,
            0.0079,
            0.0082,
            0.0087,
            0.0093,
            0.0095,
        ]
    )

    # Creating curves for interest rates and volatility
    domestic_curve = Curve(tenors, domestic_rates)
    foreign_curve = Curve(tenors, foreign_rates)
    atm_vols = VolCurve(tenors, vol_atm)
    put_vols = VolCurve(tenors, vol_25d_put)
    call_vols = VolCurve(tenors, vol_25d_call)
    df_diffs = domestic_curve.discount_factors - foreign_curve.discount_factors

    vanna_volga = VannaVolga(
        spot,
        domestic_curve,
        foreign_curve,
        atm_vols,
        put_vols,
        call_vols
    )

    df = pd.DataFrame(vanna_volga.vol_surface)
    df.to_csv("test.csv", index=False)

    x_axis = vanna_volga.domestic_curve.time
    y_axis = vanna_volga.strikes
    z_axis = df
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("Time to Maturity")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Implied Volatility")
    vol_surface = ax.plot_surface(
        x_axis, y_axis, z_axis, linewidth=0, antialiased=False, cmap="jet"
    )
    plt.legend(loc="upper right", prop=font_prop)
    plt.show()
