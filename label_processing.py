import pandas as pd
import numpy as np

df = pd.read_csv('dataset_final.csv', index_col=False)
genres_unique = []

for j in range(df[df.columns[0]].count()):
    g = df['genres'][j]
    g = g.replace("'","")
    g = g.replace(" ","")
    g = g[1:-1]
    g = g.split(',')
    for i in range(len(g)):
        if g[i] not in genres_unique:
            genres_unique.append(g[i])

df = df.reindex(df.columns.tolist() + genres_unique, axis=1, fill_value=0)

# for i in range(df[df.columns[0]].count()):
#     g = df['genres'][i].split(", ")

# df = pd.get_dummies(df, columns=genres_unique)

# for index, row in df.iterrows():
#     for val in row.genres.split(', '):
#         if val != 'NA':
#             df.loc[index, val] = 1

# df = df.drop('genres', 1)
df.to_csv('test.csv',index=False)
print(genres_unique)

