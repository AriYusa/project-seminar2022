import pandas as pd

from metrics import wape
from models import predict_sktime, walk_forward_prediction
from outliers_detection import correct_outliers
from tbats import TBATS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
import lightgbm as lgb


def dist_predict():
    train_sales = pd.read_excel("../data/preprocessed/train_sales_dist.xlsx",
                                parse_dates=True, index_col="Period")
    test_sales = pd.read_excel("../data/preprocessed/test_sales_dist.xlsx",
                               parse_dates=True, index_col="Period")
    test_sales = test_sales[['DFU', 'Customer', 'BPV', 'Total Sell-in']]

    result = pd.DataFrame([], columns=['Period', 'DFU', 'Customer', 'Total Sell-in', 'BPV', 'predicted'])

    for customer in [2, 14, 29, 34, 18]:

        train_sales = train_sales.sort_index()
        test_sales = test_sales.sort_index()
        train = train_sales[train_sales['Customer'] == customer]
        test = test_sales[test_sales['Customer'] == customer]
        dfu = test['DFU'].unique()[0]

        if customer in [18, 34]:  # для СНГ предсказания по месяцам
            train = train.resample('MS').apply(sum)
            # т.к. агрегируем по месяцам, то первую запись в тесте нужно брать (BPV равен 0),
            # иначе июль 2021 будет и в трейне и в тесте
            test = test[1:]
            test = test.resample('MS').apply(sum)

        if customer in [18, 34]:
            freq = "M"
        if customer in [2, 14, 29]:
            freq = "W"

        if customer == 2:
            train['BPV_anomaly_delete'] = correct_outliers(train['BPV'], window=6, scale=1.96, mode='delete')
            seasonal_period = 5
            model = NaiveForecaster(sp=seasonal_period)
            test['predicted'] = predict_sktime(train, test, 'BPV_anomaly_delete', freq, model)

        if customer == 14:
            seasonal_period = 5
            model = AutoARIMA(start_p=1, d=None, start_q=0,
                              max_p=3, max_d=3, max_q=3,
                              start_P=1, D=1, start_Q=0,
                              max_P=3, max_D=3, max_Q=3,
                              sp=seasonal_period, suppress_warnings=True, stepwise=False, n_jobs=-1)
            test['predicted'] = predict_sktime(train, test, 'BPV', freq,
                                               NaiveForecaster(sp=seasonal_period))

        if customer == 29:
            train['BPV_anomaly_next'] = correct_outliers(train['BPV'], window=6, scale=1.96, mode='next')
            lags = range(1, 9)
            calc_features = {
                'on_weekmean': False,
                'on_monthmean': False,
                'on_date ': False,
                'on_monthpart': True,
                'on_diff': False
            }
            objective = 'mae'
            model = lgb.LGBMRegressor(objective=objective, random_state=31)
            _, preds = walk_forward_prediction(train, test, 'BPV_anomaly_next', 'Period', [], model, lags,
                                               calc_features)
            test['predicted'] = preds

        if customer == 18:
            test['predicted'] = 0

        if customer == 34:
            seasonal_period = 5
            model = NaiveForecaster(sp=seasonal_period)
            test['predicted'] = predict_sktime(train, test, 'BPV', freq, model)

        test = test.resample('MS').apply(sum)
        test['Customer'] = customer
        test['DFU'] = dfu


        test = test.reset_index()

        result = pd.concat([result, test], ignore_index=True)

    return result

