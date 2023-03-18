# Импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model

# Определим количество дней для прогнозирования
days_to_predict = 100

# Импорт данных ByBit
with open('data.txt', 'r') as newFile:
    datalist = []
    for item in newFile.read().split('\n'):
        if item:
            datalist.append([float(x) for x in item.split(' ')])

# преобразовать данные в формат DataFrame
df = pd.DataFrame(datalist, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
df['time'] = pd.to_datetime(df['startTime'], unit='ms')
df.set_index('time', inplace=True)
df.drop(['turnover'], axis=1, inplace=True)
df.columns = ['open', 'startTime', 'high', 'low', 'close', 'volume']
df = df.astype(float)
df = df.iloc[::-1]

# Создадим датасет для обучения модели
data = df.filter(['close'])
dataset = data.values
last_days = data.values[-days_to_predict:]
training_data_len = int(np.ceil( len(dataset) * .8 ))

# Нормализуем данные
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Создадим функцию для создания датасета
def create_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        Y.append(data[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y

# Создадим датасеты для обучения и тестирования модели
train_data = scaled_data[:training_data_len, :]
test_data = scaled_data[training_data_len - 60:, :]
x_train, y_train = create_dataset(train_data)
x_test, y_test = create_dataset(test_data)

model = load_model('model2.h5')
if (not model):
    # Создадим модель RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Сохранение модели в файл
    model.save('model2.h5')

# Сделаем прогноз на тестовом датасете
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print('Test RMSE:', rmse)

# Получим последние 60 дней из исходных данных
last_60_days = data[-60:].values

# Нормализуем последние 60 дней
last_60_days_scaled = scaler.transform(last_60_days)

# Создадим пустой список для будущих прогнозов
predicted_prices = []

# Генерация прогнозов для заданного количества дней
for i in range(days_to_predict):
    # Создадим датасет для прогнозирования
    X_test = []
    X_test.append(last_60_days_scaled[-60:])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Получим прогноз
    pred_price = model.predict(X_test)

    # Добавим прогноз в список
    predicted_prices.append(pred_price[0])

    # Добавим прогноз в последние 60 дней
    last_60_days_scaled = np.append(last_60_days_scaled, pred_price, axis=0)

    # Удалим первый элемент из последних 60 дней
    last_60_days_scaled = np.delete(last_60_days_scaled, 0, axis=0)

# Обратно масштабируем прогнозы
predicted_prices = scaler.inverse_transform(predicted_prices)


# Визуализируем исходные данные
plt.figure(figsize=(16,8))
plt.title('Bitcoin Closing Price History')
plt.plot(predicted_prices)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price USD ($)', fontsize=18)
plt.show()
