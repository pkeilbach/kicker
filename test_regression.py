import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# https://realpython.com/linear-regression-in-python/#multiple-linear-regression-with-scikit-learn
# https://scikit-learn.org/stable/install.html

df = pd.read_csv('data/ligadaten.csv')
points_cum = df.cumsum()

add_mds = 4
X = np.arange(len(points_cum) + add_mds).reshape(-1, 1)
Y = points_cum.to_numpy()
D = 3
t = 34

# transform data to polynomial
transformer = PolynomialFeatures(degree=D)
X_poly = transformer.fit_transform(X)

# fit the linear regression model on first 34 matchdays
model = LinearRegression().fit(X_poly[:t], Y[:t])

# predict on 38 matchdays
y_pred = model.predict(X_poly)
df_38 = pd.DataFrame(y_pred, columns=df.columns)

fig = go.Figure()

colors = iter(px.colors.qualitative.Plotly)

for c in df.columns:
    color = next(colors)
    fig.add_trace(go.Scatter(x=np.arange(40), y=points_cum[c], mode='markers', name=c, line=dict(color=color)))
    fig.add_trace(go.Scatter(x=np.arange(40), y=df_38[c], showlegend=False, line=dict(color=color)))

fig.show()


t = 27
# x_t = X[:t]
# Y_t = Y[:t]
# X_t = transformer.fit_transform(x_t)
model_t = LinearRegression().fit(X_poly[:t], Y[:t])
# X = np.arange(len(df)).reshape(-1, 1)
# X = transformer.fit_transform(X)
y_pred = model_t.predict(X_poly[:34])
df_t = pd.DataFrame(y_pred, columns=df.columns)

fig_1 = go.Figure()
colors = iter(px.colors.qualitative.Plotly)
for c in df.columns:
    color = next(colors)
    fig_1.add_trace(go.Scatter(x=np.arange(1, 35), y=points_cum[c].iloc[:t], mode='markers', name=c, line=dict(color=color)))
    fig_1.add_trace(go.Scatter(x=np.arange(1, 35), y=df_t[c], showlegend=False, line=dict(color=color)))

fig_1.show()

print('success')

# nach 10, 17, 20, 30 Spieltagen



