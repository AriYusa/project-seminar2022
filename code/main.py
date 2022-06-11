import pandas as pd
from metrics import wape
from data_preprocessing import preprocess_data
from distributors_predction import dist_predict
from net_prediction import net_predict
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print('Preprocessing data...')
    preprocess_data()

    print('Getting predictions for distributors...')
    dist_dataframe = dist_predict()
    print(f"Distributors WAPE {wape(dist_dataframe['BPV'], dist_dataframe['predicted'])}")

    print('Getting predictions for a trading network...')
    net_dataframe = net_predict()
    print(f"Nets WAPE {wape(net_dataframe['BPV'], net_dataframe['predicted'])}")

    result = pd.concat((dist_dataframe, net_dataframe))
    print(f"Overall WAPE {wape(result['BPV'], result['predicted'])}")
    file_dest = 'result.xlsx'
    result.to_excel(file_dest)
    print(f"Results have been saved successfully to {file_dest}!")

