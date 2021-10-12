import pandas as pd
import numpy as np

df = pd.read_csv('test.csv',index_col=False)
genres_unique = []

for j in range(df[df.columns[0]].count()):
    g = df['genres'][j]
    g = g.replace("'","")
    g = g.replace(" ","")
    g = g[1:-1]
    g = g.split(',')
    for i in range(len(g)):
        df[g[i]][j] = 1

df.to_csv('test2.csv',index=False)
print(genres_unique)

