"""ARIMA model wrapper using statsmodels."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """Thin wrapper around :class:`statsmodels.tsa.arima.model.ARIMA`.

    ARIMA is a purely generative model: it is fit on a univariate training
    series and forecasts entirely from its own internal state.  It does **not**
    consume ``X_test`` windows at inference time.  This is an important
    distinction when comparing it to RNN / LSTM in robustness experiments —
    test-time input corruption does not affect ARIMA predictions.

    Args:
        order: ``(p, d, q)`` ARIMA order.  Defaults to ``(5, 1, 0)``.
    """

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        self.order = order
        self._result = None
        self._train_len: int = 0

    def fit(self, train_series: np.ndarray) -> "ARIMAModel":
        """Fit ARIMA on a univariate training time series.

        Args:
            train_series: 1-D array of chronologically ordered observations.

        Returns:
            Self, to allow method chaining.
        """
        train_series = np.asarray(train_series, dtype=np.float64)
        self._train_len = len(train_series)
        model = ARIMA(train_series, order=self.order)
        self._result = model.fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Forecast *steps* steps beyond the end of the training series.

        Args:
            steps: Number of future time steps to forecast.

        Returns:
            1-D float32 array of length *steps*.

        Raises:
            RuntimeError: If :meth:`fit` has not been called first.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast, dtype=np.float32)
