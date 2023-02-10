import pandas as pd

from pandas import read_parquet

# data = read_parquet("../datasets/TopLandscape/test_file.parquet")
# print(data.count())

# print(data.head())

"""
parquet文件:
1、列式存储的一种文件类型
2、
"""
# data = read_parquet("../datasets/TopLandscape/val_file.parquet")
# print(data.count())

# print(data.head())


# print("val")
# data = read_parquet("../datasets/TopLandscape/val_file.parquet")
# print(data.count())

# print(data.head())

a = 20 // 3.999
print(a)

import torch

b = torch.tensor([[[1,2],[1,2]]])

print(b.shape)

import argparse

parser = argparse.ArgumentParser()

#action=‘store_true’，只要运行时该变量有传参就将该变量设为True。
parser.add_argument('--predict',action='store_true', default=False,
                    help='run prediction instead of training')


def main():
    #解析参数 
    args = parser.parse_args()
    print(args.predict)

main()



print(1 if True else 2)
print(1 if False else 2)