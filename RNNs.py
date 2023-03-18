# Импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model

predict_for = 20

# Загрузим данные
url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=YOUR_API_KEY&datatype=csv'
df = pd.read_csv(url, index_col='timestamp', parse_dates=True)

# Удаляем ненужные столбцы
df.drop(['open (USD)', 'high (USD)', 'low (USD)', 'volume'], axis=1, inplace=True)

# Изменяем названия столбцов для удобства
df.rename(columns={'close (USD)': 'close'}, inplace=True)

# Визуализируем исходные данные
# plt.figure(figsize=(16,8))
# plt.title('Bitcoin Closing Price History')
# plt.plot(df['close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Price USD ($)', fontsize=18)
# plt.show()

# Создадим датасет для обучения модели
data = df.filter(['close'])
dataset = data.values[:predict_for:-1]
training_data_len = int(np.ceil( len(dataset) * .8 ))

# Нормализуем данные
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Загрузка модели из файла
model = load_model('model.h5')
if (not model):
    # Создадим датасет для обучения модели
    train_data = scaled_data[0:training_data_len , : ]
    x_train = []
    y_train = []

    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Создадим модель RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Скомпилируем модель
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Обучим модель
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Сохранение модели в файл
    model.save('model.h5')

# Создадим тестовый датасет
test_data = scaled_data[training_data_len - 60: , : ]
x_test = []
y_test = dataset[training_data_len : , : ]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Получим прогноз цены
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# for i in range(predict_for):

print(scaled_data[-1])
print(len(scaled_data[-1]))
print('------------')
print(x_test[-1])
print(len(x_test[-1]))

# Рассчитаем RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print('rmse')

plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()