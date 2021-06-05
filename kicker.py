import pandas as pd
import numpy as np

df = pd.read_csv('data/ligadaten.csv')

# total mean
print(df.values.mean())
# max by manager
print(df.max().sort_values(ascending=False))
# min by manager
print(df.min().sort_values(ascending=True))
# mean by manager is equivalent to league ranking
print(df.mean().sort_values(ascending=False))
# league best 5 matchdays
print(df.mean(axis=1).sort_values(ascending=False).head(5))
# league worst 5 matchdays
print(df.mean(axis=1).sort_values(ascending=True).head(5))

print(df.values.std())
print(df.std().sort_values(ascending=False))
print(df.std().sort_values(ascending=False))

# which managers are "similar"
print(df.corr())

# stats
print(df.cumsum())


# match day ranks
# https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
ranks = np.argsort(np.argsort(df.to_numpy() * -1)) + 1
md_ranks = pd.DataFrame(ranks, columns=df.columns)

# cum ranks
ranks = np.argsort(np.argsort(df.cumsum().to_numpy() * -1)) + 1
cum_ranks = pd.DataFrame(ranks, columns=df.columns)

# pandas scatter matrix

print('success')
