import time
import requests

# задать параметры запроса
day = 1000*60*60*24
curr_time = round(time.time()*1000)


url = 'https://api.bybit.com/v5/market/kline'
symbol = 'BTCUSDT'
category = 'linear'
interval = '5'
end_time = curr_time
start_time = curr_time - day
limit = 200

datalist = []

for i in range(100):
    # отправить запрос и получить данные
    params = {'symbol': symbol, 'category': category, 'start': start_time, 'end': end_time, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()['result']
    end_time = end_time - day
    start_time = start_time - day
    datalist = datalist + data['list']

print(datalist)

with open('data.txt', 'w') as newFile:
    for item in datalist:
        newFile.write(' '.join(item) + '\n')