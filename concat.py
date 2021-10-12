import pandas as pd


data1 = pd.read_csv("genres2.csv")
data2 = pd.read_csv("clean4.csv",usecols=['overview'])

data3 = pd.concat([data1, data2],ignore_index=True,axis=1)

data3.to_csv("dataset_final.csv", mode="w", index=False)