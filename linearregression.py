import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# задать параметры запроса
url = 'https://api.bybit.com/v5/market/kline'
symbol = 'BTCUSDT'
category = 'linear'
interval = 'W'
limit = 200

# отправить запрос и получить данные
params = {'symbol': symbol, 'category': category, 'interval': interval, 'limit': limit}
response = requests.get(url, params=params)
data = response.json()['result']

datalist = data['list']

# преобразовать данные в формат DataFrame
df = pd.DataFrame(datalist, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
df['time'] = pd.to_datetime(df['startTime'], unit='ms')
df.set_index('time', inplace=True)
df.drop(['turnover'], axis=1, inplace=True)
df.columns = ['open', 'startTime', 'high', 'low', 'close', 'volume']
df = df.astype(float)
df = df.iloc[::-1]

# создать новую фичу, показывающую изменение цены за период
df['price_change'] = df['close'].pct_change()

# создать новый DataFrame для предсказания цены
df_predict = df.tail(1).copy()

# убрать последнюю строку, так как она является целевым значением
df = df.iloc[:-1]

# создаем объект класса SimpleImputer для замены NaN на среднее значение
imputer = SimpleImputer(strategy='mean')

# выбрать фичи для обучения модели
X = df[['open', 'high', 'low', 'volume', 'price_change']]
# заменяем NaN на среднее значение в столбце X
X = imputer.fit_transform(X)

y = df['close']

# создать и обучить модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# предсказать цену криптовалюты
predicted_price = model.predict(df_predict[['open', 'high', 'low', 'volume', 'price_change']])[0]
print('Предсказанная цена:', predicted_price)

# отобразить график предсказанной цены и реальной цены
plt.plot(df.index, df['close'])
plt.plot(df_predict.index, predicted_price)
plt.xlabel('Дата')
plt.ylabel('Цена, USD')
plt.legend(['Реальная цена', 'Предсказанная цена'])
plt.show()