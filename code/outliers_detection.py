import numpy as np
from sklearn.metrics import mean_absolute_error


def correct_outliers(data, window=12, scale=1.96, mode='next'):
    """
    Находит все значения, которые выходят за границу доверительного интервала скользящего среднего.
    Далее в зависимости от режима корректирует значения.
    :param data:
    :param window:
    :param scale:
    :param mode:
    :return:
    """
    series = data
    rolling_mean = series.rolling(window).mean()

    mae = mean_absolute_error(series[window:], rolling_mean[window:])
    deviation = np.std(series[window:] - rolling_mean[window:])
    upper_bond = rolling_mean + (mae + scale * deviation)
    add = np.where(data - upper_bond > 0, data - upper_bond, 0)
    if mode == 'next':
        # Выборос приводится к границе доверительного интервала,
        # разница между выбросом к границей прибавляется к след. наблюдению
        data = data - add + np.roll(add, 1)
    elif mode == 'delete':
        # Выборос приводится к ближайшей границе доверительного интервала
        data = data - add
    return data
