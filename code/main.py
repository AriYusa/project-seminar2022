import pandas as pd
from metrics import wape
from data_preprocessing import preprocess_data
from distributors_predction import dist_predict
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #preprocess_data()

    result = pd.DataFrame([], columns=['Period', 'DFU', 'Customer', 'Total Sell-in', 'BPV', 'predicted'])
    dist_dataframe = dist_predict()
    print(f"Distributors WAPE {wape(dist_dataframe['BPV'], dist_dataframe['predicted'])}")
