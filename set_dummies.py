import pandas as pd
import numpy as np

# df = pd.read_csv('test.csv',index_col=False)
# df = pd.read_csv('dataset_new1.csv',index_col=False)
df = pd.read_csv('cmu_dataset2.csv',index_col=False)

genres_unique = []

for j in range(df[df.columns[0]].count()):
    g = df['genre_new'][j]
    g = g.replace("'","")
    g = g.replace(" ","")
    g = g[1:-1]
    g = g.split(',')
    for i in range(len(g)):
        df[g[i]][j] = 1

# df.to_csv('test2.csv',index=False)
df.to_csv('cmu_dataset3.csv',index=False)

print(genres_unique)

