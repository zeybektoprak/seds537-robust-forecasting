"""ARIMA model wrapper using statsmodels."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """Thin wrapper around :class:`statsmodels.tsa.arima.model.ARIMA`.

    Args:
        order: The ``(p, d, q)`` order of the ARIMA model.  Defaults to
            ``(5, 1, 0)``, a common starting point for stationary series.
    """

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        self.order = order
        self._result = None

    def fit(self, train_series: np.ndarray) -> "ARIMAModel":
        """Fit the ARIMA model on a univariate time series.

        Args:
            train_series: 1-D array of training observations.

        Returns:
            Self, to allow method chaining.
        """
        model = ARIMA(train_series, order=self.order)
        self._result = model.fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Forecast *steps* steps ahead from the end of the training series.

        Args:
            steps: Number of future time steps to forecast.

        Returns:
            1-D array of length *steps* containing point forecasts.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast, dtype=np.float32)
