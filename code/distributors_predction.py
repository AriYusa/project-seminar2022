import pandas as pd


def dist_predict():
    train_sales = pd.read_excel("../data/preprocessed/train_sales_dist.xlsx",
                             parse_dates=True, index_col="Period")

    test_sales = pd.read_excel("../data/preprocessed/test_sales_dist.xlsx",
                             parse_dates=True, index_col="Period")
    test_sales = test_sales[['DFU','Customer','BPV','Total Sell-in']]

    res = []

    for customer in [2,14,29,34,18]:
        pass



