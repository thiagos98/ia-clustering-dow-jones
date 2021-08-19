# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.cluster import KMeans# implements the K-Means algorithm for clustering.
from sklearn import metrics


# %%
dji = pd.read_csv('./dow_jones_index.data', sep=',')
dji.head()


# %%
dji.drop(columns=['percent_change_price', 'percent_change_volume_over_last_wk', 'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close', 'percent_change_next_weeks_price', 'days_to_next_dividend', 'percent_return_next_dividend'], inplace=True)
dji.head()


# %%
def corrigir_campo_valor(valor):
    valor = valor.replace('$', '')
    return float(valor)


# %%
def corrigir_data(data):
    data = data[:1]
    return data


# %%
dji['open'] = dji['open'].apply(corrigir_campo_valor)
dji['high'] = dji['high'].apply(corrigir_campo_valor)
dji['low'] = dji['low'].apply(corrigir_campo_valor)
dji['close'] = dji['close'].apply(corrigir_campo_valor)
dji['date'] = dji['date'].apply(corrigir_data)


# %%
dji.info()


# %%
dji = dji.sort_values(by='date')


# %%
mes = '6'
df_mes = dji[dji['date'] == mes]


# %%
train_x = df_mes[['close', 'volume']]

train_x


# %%
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x)
x_clustered = kmeans.predict(train_x)


# %%
new_data = [[41.37, 6547812]]
y_pred = kmeans.predict(new_data)
y_pred


# %%
cluster_0_temp = []
cluster_1_temp = []
cluster_2_temp = []


# %%
for i in range(x_clustered.shape[0]):
  if x_clustered[i] == 0:
    cluster_0_temp.append([df_mes.iloc[i]['stock'], df_mes.iloc[i]['volume']])
  elif x_clustered[i] == 1:
    cluster_1_temp.append([df_mes.iloc[i]['stock'], df_mes.iloc[i]['volume']])
  elif x_clustered[i] == 2:
    cluster_2_temp.append([df_mes.iloc[i]['stock'], df_mes.iloc[i]['volume']])


# %%
cluster_0 = pd.DataFrame(cluster_0_temp,columns=['stock','volume'])
cluster_1 = pd.DataFrame(cluster_1_temp,columns=['stock','volume'])
cluster_2 = pd.DataFrame(cluster_2_temp,columns=['stock','volume'])


# %%
plt.rcParams['figure.figsize'] = [15, 15]


# %%
sns.scatterplot(data=df_mes, x='close', y='volume', hue=x_clustered, palette='coolwarm_r')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6)
plt.xlabel('close')
plt.ylabel('volume')


# %%
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="stock", y="volume", data=cluster_0, log=True)
cluster_0


# %%

sns.set_theme(style="whitegrid")
ax = sns.barplot(x="stock", y="volume", data=cluster_1, log=True)
cluster_1


# %%
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="stock", y="volume", data=cluster_2, log=True)
cluster_2


# %%



