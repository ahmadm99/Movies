import pandas as pd

df = pd.read_csv('clean4.csv', usecols=['genres'])

genres_row = []
for j in range(df[df.columns[0]].count()):
    genres = str(df['genres'][j])
    genres = genres.split(',')
    for i in range(len(genres)):
        if i % 2 == 1:
            g = genres[i].split(':')
            g = g[1][2:-2]
            if g[-1] == "'":
                g = g[0:-1]
            genres_row.append(g)
    df['genres'][j] = str(genres_row)
    genres_row.clear()

df.to_csv('genres.csv')

