import  pipreqs
from data_preprocessing import preprocess_data
from distributors_predction import dist_predict


if __name__ == '__main__':
    preprocess_data()
    dist_predict()
