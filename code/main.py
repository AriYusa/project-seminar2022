import pandas as pd
import  pipreqs

from metrics import wape
from data_preprocessing import preprocess_data
from distributors_predction import dist_predict
from net_prediction import net_predict
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #preprocess_data()

    # result = pd.DataFrame([], columns=['Period', 'DFU', 'Customer', 'Total Sell-in', 'BPV', 'predicted'])
    dist_dataframe = dist_predict()
    net_dataframe = net_predict()

    result = pd.concat((dist_dataframe, net_dataframe))
    print(f"Distributors WAPE {wape(result['BPV'], result['predicted'])}")
