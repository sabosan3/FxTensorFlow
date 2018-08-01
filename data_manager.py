import os
import pandas as pd
import numpy as np


class DataManager:
    pairs = [
        "USDJPY",
        "AUDJPY",
        "EURJPY",
        "GBPJPY",
        "NZDJPY",
        "EURUSD"
    ]

    # for rename cloumns
    columns = {
        "始値": "open",
        "高値": "max",
        "安値": "min",
        "終値": "close"
    }

    def __init__(self, dirpath="./data", target="USDJPY"):
        self.dirpath = dirpath
        self.target = target
        self.all_df = self._get_all_df()
        self.labels_df = self._get_labels_df()

    def get_inputs(self, dim: int=7):
        # calculate training data
        all_delay = self._calc_delay_data_all(self.all_df, dim)
        # separate validate data
        train_x = all_delay["2007":"2016"]
        train_y = self.labels_df.loc[train_x.index]
        val_x = all_delay["2017"]
        val_y = self.labels_df.loc[val_x.index]
        test_x = all_delay["2018"]
        test_y = self.labels_df.loc[test_x.index]
        return (
            train_x.as_matrix(),
            train_y.as_matrix(),
            val_x.as_matrix(),
            val_y.as_matrix(),
            test_x.as_matrix(),
            test_y.as_matrix()
        )

    def _get_all_df(self) -> pd.DataFrame:
        all_df = []
        for pair in self.pairs:
            # create new columns name
            new_columns = {i: pair + "-" + j for i, j in self.columns.items()}
            data_file = os.path.join(self.dirpath, pair + ".csv")
            # load fx data from csv file
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            df.rename(columns=new_columns, inplace=True)
            df.index.rename("date", inplace=True)
            # obtain new columns name
            open_str, max_str, min_str, close_str = df.columns
            all_df.append(df[close_str])

        # format all types of fx closing values to pandas DataFrame
        all_df = pd.concat(all_df, axis=1)
        # for normalizing all values
        all_df = all_df.apply(lambda x: x / np.max(x))
        # log difference
        all_df = np.log(all_df / all_df.shift())
        all_df.dropna(inplace=True)
        return all_df

    def _get_labels_df(self) -> pd.DataFrame:
        # calculate differences between previous and today closing value
        diff_target = self.all_df[self.target + "-close"].diff()
        diff_target.dropna(inplace=True)
        # make one hot labels
        labels = diff_target > 0
        labels = pd.concat([labels, ~labels], axis=1)
        labels = labels.astype(float)
        return labels

    def _calc_delay_data_all(self, log_return_df: pd.DataFrame, dim: int) -> pd.DataFrame:
        """calculate delay data for all
        """
        all_data = {}
        for col, values in log_return_df.iteritems():
            all_data[col] = self._calc_delay_data(values, dim)
        return pd.concat(
            [all_data[s] for s in self.all_df.columns],
            axis=1
        )

    @staticmethod
    def _calc_delay_data(log_return_df: pd.DataFrame, dim: int) -> pd.DataFrame:
        """calculate delay data
        """
        columns = ["before {0} day".format(i) for i in range(1, dim + 1)]
        indexes = []
        delay_data = []
        for i in range(dim, len(log_return_df)):
            indexes.append(log_return_df.index[i])
            delay = []
            for d in range(1, dim + 1):
                delay.append(log_return_df.iloc[i - d])
            delay_data.append(delay)
        delay_data = pd.DataFrame(delay_data, index=indexes, columns=columns)
        return delay_data


if __name__ == '__main__':
    dl = DataManager()
    train_x, train_y, val_x, val_y, test_x, test_y = dl.get_inputs()
    print(val_y)
