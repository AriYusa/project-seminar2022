import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima
from ThymeBoost import ThymeBoost as tb

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


# Функции предобработки предсказаний
def date_features(data):
    data['Year'] = data.Period.apply(lambda x: x.year)
    data['Quarter'] = data.Period.apply(lambda x: x.quarter)
    data['Month'] = data.Period.apply(lambda x: x.month)
    data['Day'] = data.Period.apply(lambda x: x.day)
    return data

# Подготовка ключевой метрики
def wape(y_true, y_pred):
    # если y_true нулевой, возвращаем swape
    if round(y_true.sum()) == 0:
        y_true += 1
        y_pred += 1
    return np.sum(np.abs(y_true - y_pred)) / y_true.sum()

def quality(y_true, y_pred):
    return 1 - wape(y_true, y_pred)


# Функция постобработки предсказаний
def promo_to_zero(data, prediction):
    idxs = np.where(data.promo_load_shipment > 6)[0]
    prediction[idxs] = 0

    idxs = np.where(prediction < 0)
    prediction[idxs] = 0

    return prediction

# Функция подготовки данных
def train_test_split(data: pd.DataFrame, y, n_test: int):
    """
    Разбивает датасет на обучающую и тестовую части.
    :param data: DataFrame
    :param n_test: кол-во тестовых наблюдений
    :return: обучающая и тестовая
    """
    return data.iloc[:-n_test], data.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]

# создание предсказаний
def net_predict():
    # данные трейна
    all_train = pd.read_excel('../data/preprocessed/train_sales_net.xlsx')

    # подготовка теста
    all_test = pd.read_excel('../data/preprocessed/test_sales_net.xlsx')

    # лучшие модели будут храниться здесь
    best_models = dict.fromkeys(all_test.DFU.unique(), None)

    for dfu in all_train.DFU.unique():

        # оставляем только dfu для плова, так как модель будет делать предсказание только на нем
        data = all_train[all_train.DFU == dfu].copy(deep=True)
        data.drop(['Customer', 'Total Sell-in'], axis=1, inplace=True)
        data = data[data.Period > '2018-07-01']

        # оставляем только dfu для плова, так как модель будет делать предсказание только на нем
        test_data = all_test[all_test.DFU == dfu].copy(deep=True)
        test_data.drop(['Customer', 'Total Sell-in'], axis=1, inplace=True)

        # для дальнейшего разделения датасетов запомним первую и последнюю даты обоих датасетов
        last_date_train = max(data.Period)
        first_date_test = min(test_data.Period)

        # создание дополнительных фичей
        full_data = pd.concat((data, test_data))
        full_data = date_features(full_data)

        # сбрасываем признак dfu
        full_data.drop('DFU', axis=1, inplace=True)

        # все тренировочные данные
        train = full_data[full_data.Period <= last_date_train].copy(deep=True)

        # все тестовые данные
        test = full_data[full_data.Period >= first_date_test].copy(deep=True)

        train.set_index('Period', inplace=True)
        test.set_index('Period', inplace=True)

        ex_features = ['is_promo_shipment', 'promo_shipment_left',
                       'promo_shipment_right', 'promo_load_shipment',
                       'promo_week_count', 'Quarter', 'Year', 'Month']

        models = ['lasso', 'auto_arima', 'thyme']
        results = dict.fromkeys(models, None)

        # выбор лучшей модели на кросс валидации
        for model in models:

            # максимальный размер теста для валидации с раширяющимся окном
            max_test_size = 36
            step = -4

            if model == 'lasso':

                scores_train = []
                scores_val = []

                for n in range(max_test_size, 8, step):
                    X_train, X_test, y_train, y_test = train_test_split(train.drop('BPV', axis=1), train['BPV'], n)

                    lasso = Lasso().fit(X_train[ex_features], y_train)

                    prediction_train = lasso.predict(X_train[ex_features])
                    prediction_val = lasso.predict(X_test[ex_features])

                    # постобработка
                    prediction_train = promo_to_zero(X_train, prediction_train)
                    prediction_val = promo_to_zero(X_test, prediction_val)

                    score_train = quality(y_train, prediction_train)
                    score_val = quality(y_test, prediction_val)

                    scores_train.append(score_train)
                    scores_val.append(score_val)

                # предсказания для теста
                lasso = lasso.fit(train[ex_features], train['BPV'])
                prediction_test = lasso.predict(test[ex_features])
                prediction_test = promo_to_zero(test, prediction_test)

                score_test = quality(test['BPV'], prediction_test)

                results[model] = {'train': np.mean(scores_train), 'val': np.mean(scores_val), 'test': score_test,
                                  'prediction': prediction_test}

            elif model == 'auto_arima':

                scores_train = []
                scores_val = []

                for n in range(max_test_size, 12, step):
                    X_train, X_test, y_train, y_test = train_test_split(train.drop('BPV', axis=1), train['BPV'], n)
                    arima = auto_arima(y=y_train, X=X_train[ex_features], n_fits=50, start_p=2,
                                       d=None, start_q=2, max_p=7, max_d=7, max_q=7, start_P=2,
                                       D=None, start_Q=2, max_P=7, max_D=7, max_Q=7, max_order=7)

                    prediction_train = arima.predict(X_train.shape[0], X_train[ex_features])
                    prediction_val = arima.predict(X_test.shape[0], X_test[ex_features])

                    # постобработка
                    prediction_train = promo_to_zero(X_train, prediction_train)
                    prediction_val = promo_to_zero(X_test, prediction_val)

                    score_train = quality(y_train, prediction_train)
                    score_val = quality(y_test, prediction_val)

                    scores_train.append(score_train)
                    scores_val.append(score_val)

                # предсказания для теста
                arima = arima.fit(train['BPV'], X=train[ex_features], n_fits=50, start_p=2,
                                  d=None, start_q=2, max_p=7, max_d=7, max_q=7, start_P=2,
                                  D=None, start_Q=2, max_P=7, max_D=7, max_Q=7, max_order=7)
                prediction_test = arima.predict(len(test), test[ex_features])
                prediction_test = promo_to_zero(test, prediction_test)

                score_test = quality(test['BPV'], prediction_test)

                results[model] = {'train': np.mean(scores_train), 'val': np.mean(scores_val), 'test': score_test,
                                  'prediction': prediction_test}

            elif model == 'thyme':

                scores_train = []
                scores_val = []

                for n in range(max_test_size, 12, step):
                    X_train, X_test, y_train, y_test = train_test_split(train.drop('BPV', axis=1), train['BPV'], n)
                    boosted_model = tb.ThymeBoost()
                    output = boosted_model.optimize(y_train.reset_index(drop=True),
                                                    trend_estimator=['linear', 'ses', ['linear', 'ses']],
                                                    seasonal_estimator=['fourier'],
                                                    seasonal_period=[0, 15],
                                                    global_cost=['maic', 'mse'],
                                                    exogenous_estimator=['decision_tree'],
                                                    fit_type=['global'],
                                                    exogenous=[X_train[ex_features].reset_index(drop=True)], verbose=0)

                    prediction_train = boosted_model.predict(output, len(y_train),
                                                             future_exogenous=X_train[ex_features].reset_index(
                                                                 drop=True)
                                                             )['predictions'].values

                    prediction_val = boosted_model.predict(output, len(y_test),
                                                           future_exogenous=X_test[ex_features].reset_index(drop=True)
                                                           )['predictions'].values

                    # постобработка
                    prediction_train = promo_to_zero(X_train, prediction_train)
                    prediction_val = promo_to_zero(X_test, prediction_val)

                    score_train = quality(y_train, prediction_train)
                    score_val = quality(y_test, prediction_val)

                    scores_train.append(score_train)
                    scores_val.append(score_val)

                # предсказания для теста
                output = boosted_model.optimize(train['BPV'].reset_index(drop=True),
                                                trend_estimator=['linear', 'ses', ['linear', 'ses']],
                                                seasonal_estimator=['fourier'],
                                                seasonal_period=[15],
                                                global_cost=['maic', 'mse'],
                                                exogenous_estimator=['decision_tree'],
                                                fit_type=['global'],
                                                exogenous=[train[ex_features].reset_index(drop=True)], verbose=0)

                prediction_test = boosted_model.predict(output, len(test),
                                                        future_exogenous=test[ex_features].reset_index(drop=True)
                                                        )['predictions'].values
                prediction_test = promo_to_zero(test, prediction_test)

                score_test = quality(test['BPV'], prediction_test)

                results[model] = {'train': np.mean(scores_train), 'val': np.mean(scores_val), 'test': score_test,
                                  'prediction': prediction_test}

            best_models[dfu] = results

    chosed_models = {}
    for dfu in best_models:
        model_name = None
        max_val = -float('inf')
        for name in best_models[dfu]:
            if model_name is None or max_val < (best_models[dfu][name]['val'] + best_models[dfu][name]['test']) / 2:
                model_name = name
                max_val = (best_models[dfu][name]['val'] + best_models[dfu][name]['test']) / 2
        chosed_models[dfu] = model_name

    all_test = pd.read_excel('../data/preprocessed/test_sales_net.xlsx')
    result = pd.DataFrame([], columns=['Period', 'DFU', 'Customer', 'BPV', 'Total Sell-in', 'predicted'])

    for dfu in all_test.DFU.unique():
        temp = all_test[all_test.DFU == dfu][['Period', 'DFU', 'Customer', 'BPV', 'Total Sell-in']]

        temp['predicted'] = best_models[dfu][chosed_models[dfu]]['prediction']

        result = pd.concat((result, temp))

    return result
