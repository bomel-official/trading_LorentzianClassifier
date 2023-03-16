# ===== Получение данных ===== START
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# задать параметры запроса
url = 'https://api.bybit.com/v5/market/kline'
symbol = 'BTCUSDT'
category = 'linear'
interval = '15'
start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
start_time_str = start_time.isoformat()
limit = 1000

# отправить запрос и получить данные
params = {'symbol': symbol, 'category': category, 'interval': interval, 'from': start_time_str, 'limit': limit}
response = requests.get(url, params=params)
data = response.json()['result']

datalist = data['list']

# преобразовать данные в формат DataFrame
df = pd.DataFrame(datalist, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
df = df.astype(float)
df['time'] = pd.to_datetime(df['startTime'], unit='ms')
df.set_index('time', inplace=True)
df.drop(['startTime', 'volume', 'turnover'], axis=1, inplace=True)
df.columns = ['open', 'high', 'low', 'close']

# ===== Получение данных ===== END


def lorentzian_func(x, A, gamma, x0):
    return (A*gamma**2)/((x-x0)**2+gamma**2)

def fit_lorentzian(xdata, ydata):
    popt, _ = curve_fit(lorentzian_func, xdata, ydata, p0=[max(ydata), 1, xdata.mean()])
    return popt

class LorentzianClassifier:
    def __init__(self, n_bins, threshold):
        self.n_bins = n_bins
        self.threshold = threshold
        self.centers = []
        self.widths = []
        
    def fit(self, data):
        hist, bin_edges = np.histogram(data, bins=self.n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        popt = fit_lorentzian(bin_centers, hist)
        self.centers.append(popt[2])
        self.widths.append(popt[1])
        
    def predict(self, data):
        pred = []
        for val in data:
            dist = [np.abs(val-c) for c in self.centers]
            closest_center = self.centers[np.argmin(dist)]
            closest_width = self.widths[np.argmin(dist)]
            if np.abs(val - closest_center) < closest_width * self.threshold:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)

# Подготовка данных для анализа
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Создание и обучение модели Lorentzian Classification
n_bins = 50
threshold = 2
lc = LorentzianClassifier(n_bins=n_bins, threshold=threshold)
lc.fit(scaled_data)

# Получение предсказаний модели
predictions = lc.predict(scaled_data)

# Визуализация результатов
import matplotlib.pyplot as plt

plt.plot(df.index, df['close'].values)
plt.plot(df.index, scaler.inverse_transform(predictions)[:,3], 'r')
plt.show()