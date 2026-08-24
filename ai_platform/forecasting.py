"""Forecasting AI (spec Part 15).

Classical statistical forecasting - linear trend regression and simple
exponential smoothing - computed with plain arithmetic, never asked of
the LLM (the spec is explicit: "Do not ask the LLM to invent numerical
forecasts"). Confidence comes from backtested error (MAE/RMSE on held-out
history), not a made-up percentage.
"""

import statistics
from dataclasses import dataclass, field


@dataclass
class ForecastResult:
    method: str
    history: list
    forecast: list
    periods_ahead: int
    mae: float          # backtested mean absolute error
    rmse: float          # backtested root mean squared error
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__


def _linear_trend_fit(values):
    """Ordinary least squares fit of value ~ index. Returns (slope, intercept)."""
    n = len(values)
    xs = list(range(n))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    slope = num / den if den != 0 else 0.0
    intercept = y_mean - slope * x_mean
    return slope, intercept


def linear_trend_forecast(history, periods_ahead: int):
    if len(history) < 2:
        raise ValueError("linear_trend_forecast requires at least 2 historical points")

    slope, intercept = _linear_trend_fit(history)
    n = len(history)
    forecast = [slope * (n + i) + intercept for i in range(periods_ahead)]

    # Backtest: fit on all-but-last point, predict the held-out last point.
    if n >= 3:
        s2, b2 = _linear_trend_fit(history[:-1])
        predicted_last = s2 * (n - 1) + b2
        mae = abs(predicted_last - history[-1])
        rmse = mae  # single held-out point: MAE and RMSE coincide
    else:
        mae = rmse = float("nan")

    return ForecastResult(
        method="linear_trend", history=history, forecast=forecast,
        periods_ahead=periods_ahead, mae=round(mae, 4) if mae == mae else None,
        rmse=round(rmse, 4) if rmse == rmse else None,
        params={"slope": round(slope, 6), "intercept": round(intercept, 6)},
    )


def exponential_smoothing_forecast(history, periods_ahead: int, alpha: float = 0.4):
    if len(history) < 2:
        raise ValueError("exponential_smoothing_forecast requires at least 2 points")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    def smooth(series):
        level = series[0]
        levels = [level]
        for v in series[1:]:
            level = alpha * v + (1 - alpha) * level
            levels.append(level)
        return levels

    levels = smooth(history)
    last_level = levels[-1]
    # Flat forecast at the last smoothed level (standard simple-ES behavior;
    # no trend/seasonality component - this is a deliberate simplicity choice,
    # not a limitation hidden from the caller).
    forecast = [last_level] * periods_ahead

    # Backtest: one-step-ahead prediction error across the series.
    errors = []
    for i in range(1, len(history)):
        pred = levels[i - 1]
        errors.append(abs(pred - history[i]))
    mae = statistics.mean(errors) if errors else None
    rmse = (statistics.mean(e ** 2 for e in errors)) ** 0.5 if errors else None

    return ForecastResult(
        method="exponential_smoothing", history=history, forecast=forecast,
        periods_ahead=periods_ahead,
        mae=round(mae, 4) if mae is not None else None,
        rmse=round(rmse, 4) if rmse is not None else None,
        params={"alpha": alpha, "last_level": round(last_level, 4)},
    )


def forecast(history, periods_ahead: int = 3, method: str = "linear_trend", **kwargs):
    if method == "linear_trend":
        return linear_trend_forecast(history, periods_ahead)
    if method == "exponential_smoothing":
        return exponential_smoothing_forecast(history, periods_ahead, **kwargs)
    raise ValueError(f"Unknown forecasting method '{method}'. Use 'linear_trend' or 'exponential_smoothing'.")
