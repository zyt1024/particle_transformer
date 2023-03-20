from torch.utils.data import IterableDataset
import pandas as pd


class PandasIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.data_iter = pd.read_csv(file_path, iterator=True, header=None, chunksize=1)

    def __iter__(self):
        for data in self.data_iter:
            yield data


if __name__ == '__main__':
    dataset = PandasIterableDataset('test_csv.csv')
    for data in dataset:
        print(data)
