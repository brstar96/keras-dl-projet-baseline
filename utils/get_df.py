import os
import pandas as pd


def get_df_from_config(config):
    train_csv: str = os.path.join(config.data.base_path, config.data.train_csv)
    test_csv: str = os.path.join(config.data.base_path, config.data.test_csv)

    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        return train_df, test_df
    except FileNotFoundError as e:
        print('csv file not found!')
        exit(-1)
