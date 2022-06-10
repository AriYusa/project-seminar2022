import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple
from metrics import *


def cal_new_cols(data: pd.DataFrame, target_column: str, date_column: str, lags: Iterable[int], n_test: int,
                 calc_features: Dict[str, bool]) -> pd.DataFrame:
    """
    Расчитывает лаги и другие признаки.
    :param data: DataFrame, содержащий столбец с датой
    :param target_column: столбец, данные которого используются для создания лагов
    :param date_column: столбец с датой
    :param lags: список временных лагов
    :param n_test: размер тестовых данных
    :param calc_features: словарь, где ключи-параметры, значения-True/False. True означает,
    что признак будет включён в множество признаков используемых для предсказания
    :return: DataFrame с новыми столбцами
    """
    on_weekmean = calc_features.get('on_weekmean', False)
    on_monthmean = calc_features.get('on_monthmean', False)
    on_date = calc_features.get('on_date ', False)
    on_monthpart = calc_features.get('on_monthpart', False)
    on_diff = calc_features.get('on_diff', False)

    # расчёт лагов
    for n in lags:
        data[f'lag_{n}'] = data[target_column].shift(periods=n)

    # расчёт средних значений по номеру недели в году
    if on_weekmean:
        data['Week_Number'] = pd.to_datetime(data[date_column]).dt.isocalendar().week
        week_means = data.iloc[:-n_test].groupby('Week_Number')[target_column].mean()
        if 53 not in week_means.keys():
          week_means[53] = week_means[1]
        data['Week_mean'] = data['Week_Number'].apply(lambda w_n: week_means[w_n])
        if not on_date:
            data = data.drop('Week_Number', axis=1)

    # расчёт средних значений по месяцу
    if on_monthmean:
        data['Month'] = data[date_column].dt.month
        month_means = data.iloc[:-n_test].groupby('Month')[target_column].mean()
        data['Month_mean'] = data['Month'].apply(lambda w_n: month_means[w_n])
        if not on_date:
            data = data.drop('Month', axis=1)

    # расчёт средних значений по части месяца (месяц делится на 4 части)
    if on_monthpart:
        data['Month_part'] = ((data[date_column].dt.day / data[date_column].dt.days_in_month) // 0.2501).values
        part_means = data.iloc[:-n_test].groupby('Month_part')[target_column].mean()
        data['Partly_mean'] = data['Month_part'].apply(lambda w_n: part_means[w_n])
        if not on_date:
            data = data.drop('Month_part', axis=1)

    # расчёт разностей между 3я предыдущими наблюдениями
    if on_diff:
        for n in range(2):
            data[f'diff_{n + 1}_{n + 2}'] = (data[f'lag_{n + 1}'] - data[f'lag_{n + 2}']) / data[f'lag_{n + 1}']
            data.replace([np.inf, -np.inf], 0, inplace=True)

    # расчёт данных о дате
    if on_date:
        data['Year'] = data[date_column].dt.year
        data['Month'] = data[date_column].dt.month
        try:
            if pd.infer_freq(data[target_column]) not in ['M', 'MS']:
                data['Week_Number'] = data[date_column].dt.week
        except TypeError:
            pass

    return data


def train_test_split(data: pd.DataFrame, n_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датасет на обучающую и тестовую части.
    :param data: DataFrame
    :param n_test: кол-во тестовых наблюдений
    :return: обучающая и тестовая
    """
    return data.iloc[:-n_test], data.iloc[-n_test:]


def walk_forward_prediction(train: pd.DataFrame, test: pd.DataFrame, target_column: str, date_column: str,
                            features: list, model, lags: Iterable[int], calc_features: Dict[str, bool]):
    """
    Получение предсказания путём итеративного предсказания на один шаг вперёд. Причина: при вычилениии лагов,
    в тестовых данных возникают лики, для предотвращения этой ситуации
    лаги в тестовых данных перечитываются с учетом предсказанных значений, а не истинных.
    :param train: обучабщая выборка
    :param test: тестовая выборка
    :param target_column: название целевой колонки, содержащей значения, про которым будут формироваться лаги
    :param date_column: название колонки, содержащей дату
    :param features: список дополнительных признаков
    :param model: модель, используемая для прогнозирования
    :param lags: список временных лагов
    :param calc_features: словарь, где ключи-параметры, значения-True/False. True означает,
    что признак будет включён в множество признаков используемых для предсказания
    :return: истинные значения и предсказанные значения тестовой выборки
    """
    train_data = train.copy()
    test_data = test.copy()

    if (target_column != 'BPV') and ('BPV' in train_data.columns):
        train_data = train_data.drop('BPV', axis=1)
        train_data = train_data.rename({target_column: 'BPV'}, axis=1)

    trues = test_data['BPV'].values.copy()
    test_data['BPV'] = None
    n_test = len(test_data)

    # обучающая и тестовая выборка соединяются для последющего подсчёта лагов
    data = pd.concat([train_data[['BPV'] + features], test_data[['BPV'] + features]])

    temp_preds = data['BPV'].values
    predictions = []

    data = cal_new_cols(data.reset_index(), 'BPV', date_column, lags, n_test, calc_features)
    train_data, test_data = train_test_split(data, n_test)

    trainX, trainy = train_data.drop('BPV', axis=1), train['BPV']
    # fit model
    model.fit(trainX.drop(date_column, axis=1).values, trainy.values)

    # итеративный прогноз
    for i in range(n_test):
        testX, testy = test_data.drop('BPV', axis=1), test['BPV']
        # получение предсказания для объекта и тестовой выборки
        yhat = model.predict(testX.drop(date_column, axis=1).values)[i]
        predictions.append(yhat)

        # обновляем значение
        temp_preds[data.shape[0] - n_test + i] = yhat
        data['BPV'] = temp_preds

        # перерасчитываем признаки с учетом предсказанного значения
        data = cal_new_cols(data, 'BPV', date_column, lags, n_test, calc_features)
        train_data, test_data = train_test_split(data, n_test)

    return trues, predictions


def extending_window_cv(data: pd.DataFrame, target_column: str, date_column: str, features: List, model,
                        lags: Iterable[int], calc_features: Dict[str, bool], max_test_size: int):
    """
    Валидация расширяющимся окном.
    :param data: обучающая выборка
    :param target_column: название целевой колонки, содержащей значения, про которым будут формироваться лаги
    :param date_column: название колонки, содержащей дату
    :param features: список дополнительных признаков
    :param model: модель, используемая для прогнозирования
    :param lags: список временных лагов
    :param calc_features: словарь, где ключи-параметры, значения-True/False. True означает,
    что признак будет включён в множество признаков используемых для предсказания
    :param max_test_size: максимальный размер отложенной выборки
    :return: среднее значение метрики на валидации
    """
    scores = []
    # постепенно уменьшается размер отложенной выборки
    for n in range(max_test_size, 0, -1):
        train, test = train_test_split(data, n)
        trues, predictions = walk_forward_prediction(train, test, target_column, date_column, features, model, lags,
                                                     calc_features)
        score = wape(*resample_monthly(test['BPV'], predictions))
        scores.append(score)
    return np.array(scores).mean()
