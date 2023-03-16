# ===== Получение данных ===== START
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# задать параметры запроса
url = 'https://api.bybit.com/v5/market/kline'
symbol = 'BTCUSDT'
category = 'linear'
interval = '15'
start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
start_time_str = start_time.isoformat()
limit = 200

# отправить запрос и получить данные
params = {'symbol': symbol, 'category': category, 'interval': interval, 'from': start_time_str, 'limit': limit}
response = requests.get(url, params=params)
data = response.json()['result']

datalist = data['list']

# преобразовать данные в формат DataFrame
df = pd.DataFrame(datalist, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
df['time'] = pd.to_datetime(df['startTime'], unit='ms')
df.set_index('time', inplace=True)
df.drop(['startTime', 'turnover'], axis=1, inplace=True)
df.columns = ['open', 'high', 'low', 'close', 'volume']
df = df.astype(float)

# ===== Получение данных ===== END

# Выберем признаки для модели
X = df[['open', 'high', 'low', 'volume']]
y = df['close']

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создадим модель линейной регрессии
model = LinearRegression()

# Обучим модель на обучающей выборке
model.fit(X_train, y_train)

# Сделаем предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Оценим качество модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R^2 Score:', r2)