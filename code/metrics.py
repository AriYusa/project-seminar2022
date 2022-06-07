import numpy as np
import pandas as pd
from typing import Union, List


def wape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates WAPE (Weighted Absolute Percentage Error)
    :param y_true: array of true values
    :param y_pred: array of predicted values
    :return: value of error
    """
    return np.sum(np.abs(y_true - y_pred)) / y_true.sum()


def quality(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates quality of prediction as 1-WAPE
    :param y_true: array of true values
    :param y_pred: array of predicted values
    :return: value of quality
    """
    return 1 - wape(y_true, y_pred)


def resample_monthly(y_true: pd.Series, y_pred: Union[np.ndarray, pd.Series, List]):
    """
    Resamples arrays to monthly frequency
    :param y_true: pd.Series object with pd.DatetimeIndex
    :param y_pred: array of predicted values
    :return: resampled array with true values, resampled array with predicted values
    """
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise TypeError(f"Type of index must be pd.DatetimeIndex, but got {type(y_true.index)}")

    y_true = y_true.resample('MS').apply(sum)
    y_pred = pd.Series(y_pred, index=y_true.index).resample('MS').apply(sum)
    return y_true, y_pred
