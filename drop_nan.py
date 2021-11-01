import ast

import pandas as pd
import re
df = pd.read_csv('dataset_new1.csv', index_col=False)
df.genres = df.genres.apply(ast.literal_eval)

# for i in range(df[df.columns[0]].count()):
#     row = df['genres'][i]
#     df.at[i, 'genres'] = ast.literal_eval(row)

df.to_csv('dataset_new2.csv', index=False)
# # OVERVIEW
# df = pd.read_csv('clean.csv', usecols=['overview'])
# skip = []
#
# for i in range(df[df.columns[0]].count()):
#     if len(df['overview'][i]) == 7:
#         skip.append(i+1)
# df = pd.read_csv("clean.csv", skiprows=skip)
# df.to_csv('clean2.csv', mode="w", index=False)
#
#
# # GENRES
# df = pd.read_csv('clean2.csv', usecols=['genres'])
# skip = []
#
# for i in range(df[df.columns[0]].count()):
#     if df['genres'][i] == '[]':
#         skip.append(i+1)
# df = pd.read_csv("clean2.csv", skiprows=skip)
# df.to_csv('clean4.csv', mode="w", index=False)


# # GENRES
# df = pd.read_csv('genres.csv', usecols=['genres'])
# skip = []
#
# for j in range(df[df.columns[0]].count()):
#     if any(i.isdigit() for i in df['genres'][j]):
#         skip.append(j+1)
# df = pd.read_csv("genres.csv", skiprows=skip)
# df.to_csv('genres2.csv', mode="w", index=False)



