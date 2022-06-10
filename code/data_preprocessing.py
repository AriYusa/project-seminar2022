import datetime
import pandas as pd
import xlrd
import openpyxl
import os


# функция, заполняющая пропущенные периоды, в которые не было продаж
def fill_dates(data, fr=None, up=None):
    result = pd.DataFrame()

    for cust in data.Customer.unique():
        for dfu in data[data.Customer == cust].DFU.unique():
            temp = data[(data['Customer'] == cust) & (data['DFU'] == dfu)]

            if fr is not None and pd.Timestamp(fr) not in temp.Period.values:
                temp = pd.concat((pd.DataFrame([[dfu, cust, pd.Timestamp(fr), 0, 0]], columns=temp.columns), temp))

            if up is not None and pd.Timestamp(up) not in temp.Period.values:
                temp = pd.concat((temp, pd.DataFrame([[dfu, cust, pd.Timestamp(up), 0, 0]], columns=temp.columns)))

            temp = temp.set_index('Period').asfreq(freq=pd.DateOffset(days=7)).fillna(
                value={"DFU": dfu, "Customer": cust, "BPV": 0, "Total Sell-in": 0, "Year": 0, "Month": 0, "Day": 0})
            result = pd.concat((
                result,
                temp
            ))
    return result.reset_index()


# функция, зануляющая возвраты по BPV и Total Sell-in, если таковые имеются
def fill_returns(data):
    result = pd.DataFrame()

    for cust in data.Customer.unique():
        for dfu in data[data.Customer == cust].DFU.unique():
            temp = data[(data.Customer == cust) & (data.DFU == dfu)]

            while (temp['BPV'] < 0).any():

                if temp.iloc[0].BPV < 0:
                    temp.loc[temp.Period == temp.iloc[0].Period, 'BPV'] = 0

                idx_neg = temp[temp['BPV'] < 0].index

                temp.loc[idx_neg - 1, 'BPV'] = temp.loc[idx_neg - 1, 'BPV'] + temp.loc[idx_neg, 'BPV'].values
                temp.loc[idx_neg, 'BPV'] = 0

            while (temp['Total Sell-in'] < 0).any():

                if temp.iloc[0]['Total Sell-in'] < 0:
                    temp.loc[temp.Period == temp.iloc[0].Period, 'Total Sell-in'] = 0

                idx_neg = temp[temp['Total Sell-in'] < 0].index

                temp.loc[idx_neg - 1, 'Total Sell-in'] = temp.loc[idx_neg - 1, 'Total Sell-in'] + temp.loc[
                    idx_neg, 'Total Sell-in'].values
                temp.loc[idx_neg, 'Total Sell-in'] = 0

            result = pd.concat((result, temp))

    return result


# функция соединяющая таблицы promo и sales
def join_promo(data, promo):
    result = pd.DataFrame()

    for dfu in data.DFU.unique():

        temp_sales = data[data.DFU == dfu]
        temp_promo = promo[promo.DFU == dfu]

        ### binary
        is_promo_shipment = []
        promo_shipment_right = []
        promo_shipment_left = []
        ### numeral
        promo_load_shipment = []
        promo_week_count = []

        # словарь для подсчета кол-ва использования промо (надо проверить данный признак)
        promo_count = dict()

        for idx in temp_sales.index:

            # создаем все необходимые переменные для простановки промо от sales
            period_start = temp_sales.loc[idx, 'Period']
            period_end = temp_sales.loc[idx, 'Period'] + datetime.timedelta(days=6)

            # условие для отбора промо
            condition_shipment_full = (
                        ((temp_promo.Start_date < period_start) & (temp_promo.End_date > period_start)) & (
                            (temp_promo.Start_date < period_end) & (temp_promo.End_date > period_end)))
            condition_shipment_right = (
                        (~((temp_promo.Start_date < period_start) & (temp_promo.End_date > period_start))) & (
                            (temp_promo.Start_date < period_end) & (temp_promo.End_date > period_end)))
            condition_shipment_left = (
                        ((temp_promo.Start_date < period_start) & (temp_promo.End_date > period_start)) & (
                    ~((temp_promo.Start_date < period_end) & (temp_promo.End_date > period_end))))

            # получаем списки промо попадающие под выставленные условия
            shipment_promo_full = temp_promo[condition_shipment_full]
            shipment_promo_right = temp_promo[condition_shipment_right]
            shipment_promo_left = temp_promo[condition_shipment_left]

            # проверим длину всех таблиц
            ## shipment
            if shipment_promo_full.shape[0] == 0 and shipment_promo_right.shape[0] == 0 and shipment_promo_left.shape[
                0] == 0:
                is_promo_shipment.append(0)
                promo_shipment_right.append(0)
                promo_shipment_left.append(0)
                promo_load_shipment.append(0)
                promo_week_count.append(0)

            elif shipment_promo_full.shape[0] != 0:
                # получаем полное промо
                shipment_promo_full['Length'] = shipment_promo_full['End_date'] - shipment_promo_full['Start_date']
                promo_full = shipment_promo_full.sort_values(by='Length').iloc[0]

                # promo count
                promo_id = str(promo_full['Start_date'].date()) + '-' + str(promo_full['End_date'].date())
                if promo_id not in promo_count:
                    promo_count[promo_id] = 1
                    promo_week_count.append(promo_count[promo_id])
                else:
                    promo_count[promo_id] += 1
                    promo_week_count.append(promo_count[promo_id])

                # main features
                is_promo_shipment.append(1)
                promo_shipment_right.append(0)
                promo_shipment_left.append(0)
                promo_load_shipment.append(7)

            else:

                is_promo_shipment.append(1)
                promo_load_shipment.append(0)
                promo_week_count.append(0)

                if shipment_promo_left.shape[0] != 0:
                    # получаем правое промо
                    shipment_promo_left['Length'] = shipment_promo_left['End_date'] - shipment_promo_left['Start_date']
                    promo_left = shipment_promo_left.sort_values(by='Length').iloc[0]

                    # promo count
                    promo_id = str(promo_left['Start_date'].date()) + '-' + str(promo_left['End_date'].date())
                    if promo_id not in promo_count:
                        promo_count[promo_id] = 1
                        promo_week_count[-1] = promo_count[promo_id]
                    else:
                        promo_count[promo_id] += 1
                        promo_week_count[-1] = promo_count[promo_id]

                    # main features
                    promo_left_start = promo_left['Start_date']
                    promo_left_end = promo_left['End_date']

                    promo_load_shipment[-1] += (promo_left_end - period_start).days

                    promo_shipment_left.append(1)

                else:
                    promo_shipment_left.append(0)

                if shipment_promo_right.shape[0] != 0:
                    # получаем правое промо
                    shipment_promo_right['Length'] = shipment_promo_right['End_date'] - shipment_promo_right[
                        'Start_date']
                    promo_right = shipment_promo_right.sort_values(by='Length').iloc[0]

                    # promo count
                    promo_id = str(promo_right['Start_date'].date()) + '-' + str(promo_right['End_date'].date())
                    if promo_id not in promo_count:
                        promo_count[promo_id] = 1
                        promo_week_count[-1] = promo_count[promo_id]
                    else:
                        promo_count[promo_id] += 1
                        promo_week_count[-1] = promo_count[promo_id]

                    # main features
                    promo_right_start = promo_right['Start_date']
                    promo_right_end = promo_right['End_date']

                    promo_load_shipment[-1] += (period_end - promo_right_start).days + 1

                    promo_shipment_right.append(1)

                else:
                    promo_shipment_right.append(0)

                if promo_load_shipment[-1] > 7:
                    promo_load_shipment[-1] = 7

        # присоединяем размеченные данные
        temp_sales['is_promo_shipment'] = is_promo_shipment
        temp_sales['promo_shipment_left'] = promo_shipment_left
        temp_sales['promo_shipment_right'] = promo_shipment_right
        temp_sales['promo_load_shipment'] = promo_load_shipment

        # присоединяем фичу promo count
        temp_sales['promo_week_count'] = promo_week_count

        result = pd.concat((result, temp_sales))

    return result


def preprocess_data():
    ### Promo preprocessing
    # загрузка данных по промо
    train_promo = pd.read_excel('../data/train_promo.xlsx')
    test_promo = pd.read_excel('../data/test_promo.xlsx')
    # оставляем данные только по промо для кастомера 1
    train_promo = train_promo[train_promo.Customer == 1]
    test_promo = test_promo[test_promo.Customer == 1]
    # переимнование дфу
    train_promo.loc[train_promo.DFU == 'Рис длиннозерный 500 гр', 'DFU'] = 'Рис длиннозерный 486 гр'
    test_promo.loc[test_promo.DFU == 'Рис длиннозерный 500 гр', 'DFU'] = 'Рис длиннозерный 486 гр'
    # переименование колонок для большего удобства
    train_promo.rename(
        columns={'Promo №': 'Promo_', 'Start Date on shelf': 'Start_date_shelf', 'End Date on shelf': 'End_date_shelf',
                 'Promo Days on shelf': 'Shelf', 'Shipment days to promo start': 'Ship_days_to_promo',
                 'First Date of shipment': 'Start_date', 'End Date of shipment': 'End_date',
                 'Shipment duration': 'Ship_duration', 'Discount, %': 'Discount', 'Units SoD': 'SoD_promo'},
        inplace=True)

    test_promo.rename(
        columns={'Promo №': 'Promo_', 'Start Date on shelf': 'Start_date_shelf', 'End Date on shelf': 'End_date_shelf',
                 'Promo Days on shelf': 'Shelf', 'Shipment days to promo start': 'Ship_days_to_promo',
                 'First Date of shipment': 'Start_date', 'End Date of shipment': 'End_date',
                 'Shipment duration': 'Ship_duration', 'Discount, %': 'Discount', 'Units SoD': 'SoD_promo'},
        inplace=True)

    ### Sales preprocessing
    # загрузка данных по продажам
    train_sales = pd.read_excel('../data/train_sales.xlsx')
    test_sales = pd.read_excel('../data/test_sales.xlsx')
    # заполнение продаж для пропущенных периодов по всем дфу и кастомерам
    train_sales = fill_dates(train_sales, fr=None, up=str(max(train_sales.Period).date()))
    train_sales = train_sales.convert_dtypes()
    test_sales = fill_dates(test_sales, fr=str(min(test_sales.Period).date()), up=str(max(test_sales.Period).date()))
    test_sales = test_sales.convert_dtypes()
    # зануление возвратов продаж путем вычитания из прошлых периодов
    train_sales = fill_returns(train_sales)
    test_sales = fill_returns(test_sales)

    # разбиение данных на данные для сети и дистрибутора
    # данные для сети
    train_sales_net = train_sales[train_sales.Customer == 1]
    test_sales_net = test_sales[test_sales.Customer == 1]
    # переимнование дфу
    train_sales_net.loc[train_sales_net.DFU == 'Рис длиннозерный 500 гр', 'DFU'] = 'Рис длиннозерный 486 гр'
    # оставляем только те дфу, которые есть и в трейне, и в тесте
    dfu_list = set(train_sales_net.DFU.unique()).intersection(set(test_sales_net.DFU.unique()))
    train_sales_net = train_sales_net[train_sales_net.DFU.isin(dfu_list)]
    test_sales_net = test_sales_net[test_sales_net.DFU.isin(dfu_list)]
    # объединение таблиц promo и sales
    train_sales_net = join_promo(train_sales_net, train_promo)
    test_sales_net = join_promo(test_sales_net, test_promo)
    # данные для дистрибутора
    train_sales_dist = train_sales[train_sales.Customer != 1]
    test_sales_dist = test_sales[test_sales.Customer != 1]

    ## Выгрузка данных

    if not os.path.exists('../data/preprocessed'):
        os.makedirs('../data/preprocessed')
    # Для сети
    train_sales_net.to_excel('../data/preprocessed/train_sales_net.xlsx')
    test_sales_net.to_excel('../data/preprocessed/test_sales_net.xlsx')

    # Для дистрибутора
    train_sales_dist.to_excel('../data/preprocessed/train_sales_dist.xlsx')
    test_sales_dist.to_excel('../data/preprocessed/test_sales_dist.xlsx')