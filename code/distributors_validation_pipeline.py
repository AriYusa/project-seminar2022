import itertools
import pandas as pd
import lightgbm as lgb
from outliers_detection import correct_outliers
from models import extending_window_cv

# cчитываем обучающие данные
train_sales = pd.read_excel("/data/ПЕ_train_sales.xlsx", parse_dates=True, index_col="Period")

# запускаем процесс перебора параметов
customers = [2, 14, 29, 34, 18]
for customer in customers:
    results = []
    # множество параметров и их значений
    grid_configs = {
        'tseries': ['orig_series', 'corrected_series'],
        'target_columns': ['BPV', 'corrected_BPV', 'corrected_BPV_d'],
        'features_configs': itertools.product([True, False], repeat=5),
        'objectives': ['mae', 'mse'],
        'lags': [range(1, 6), range(1, 9), range(1, 13)]
    }

    combin = itertools.product(*list(grid_configs.values()))
    for i, c in enumerate(combin):
        tseries_mode, target_column, (on_weekmean, on_monthmean, on_date, on_monthpart, on_diff), objective, lags = c

        # для каждого клиента строим отдельную модель
        train = train_sales[train_sales['Customer'] == customer][['BPV']]

        if customer in [18, 34]:  # для СНГ предсказания по месяцам
            train = train.resample('MS').apply(sum)

        if customer in [18, 34]:
            window = 12  # год для помесячных (18 и 34 клиенты)
            test_size = 9  # 9 месяцев
        if customer in [2, 14, 29]:
            window = 26  # 6 месяцев для понедельных (2,14,29 клиенты)
            test_size = round(52 / 12 * 9)  # 9 месяцев

        if customer in [18, 34]:
            freq = "M"
        if customer in [2, 14, 29]:
            freq = "W"

        # попробуем предобработать данные
        if tseries_mode == 'corrected_series':
            if customer == 2:
                # для клиента 2 уменьшить BPV до пандемии чтобы примерно сравнять с объемами продаж после пандемии
                train.loc[train.index <= pd.Timestamp("2020-02-24"), 'BPV'] = train.loc[train.index <= pd.Timestamp(
                    "2020-02-24"), 'BPV'] / 2
            elif customer == 29:
                # для клиента 29 убрать из обучающей выборки данные за 2019 год
                # в связи с заначительно большим объёмом продаж по сравнению с другими периодами
                train['BPV_gap'] = train['BPV']
                train.loc[train.index < pd.Timestamp("2020-01-01"), 'BPV_gap'] = train['BPV'].shift(52)

                train = train[['BPV_gap']].rename({'BPV_gap': 'BPV'}, axis=1)
                train = train.dropna()
            else:
                continue

        # откорректируем выбросы
        train['corrected_BPV'] = correct_outliers(train['BPV'], window=6, scale=1.96, mode='next')
        train['corrected_BPV_d'] = correct_outliers(train['BPV'], window=6, scale=1.96, mode='delete')

        if customer in [34, 18]:
            on_weekmean, on_monthpart = False, False

        calc_features = {
            'on_weekmean': on_weekmean,
            'on_monthmean': on_monthmean,
            'on_date ': on_date,
            'on_monthpart': on_monthpart,
            'on_diff': on_diff
        }

        model = lgb.LGBMRegressor(objective=objective, random_state=31)

        # запустим валидацию расширяющимся окном
        score = extending_window_cv(train, target_column, 'Period', [], model, lags, calc_features, test_size)

        results.append(
            [customer, tseries_mode, target_column, objective, f'{lags}', on_weekmean, on_monthmean, on_date,
             on_monthpart,
             on_diff, score])

        if i % 50 == 0:
            print(f"{i} combinations have been proceeded")

    res_df = pd.DataFrame(results,
                          columns=['customer', 'tseries_mode', 'target_column', 'objective', 'lags', 'on_weekmean',
                                   'on_monthmean', 'on_date', 'on_monthpart', 'on_diff', 'score'])

    res_df = res_df.drop_duplicates()

    # сохраняем результаты по каждому клиенту
    res_df.to_csv(f'val_grid_boosting_customer_{customer}.csv')

# объединяем результаты по каждому клиенту
total_results = []
for customer in [2, 14, 29, 18, 34]:
    cust_res_df = pd.read_csv(f'val_grid_boosting_customer_{customer}.csv')
    total_results.append(cust_res_df)
    frame = pd.concat(total_results, axis=0, ignore_index=True)
    frame.to_csv(f'val_grid_boosting v5.csv', index=False)
