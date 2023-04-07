import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Kỹ thuật này để khi mọi người dùng, không cần phải đổi tên đường dẫn 
dir_p = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(dir_p, '..', 'data', 'measData_ver2.csv')

# Đọc file
df = pd.read_csv(path_to_data)

def simple_moving_average(data, n):
    temp_data = np.full(len(data), np.nan)
    for i in range(n, len(data)):
        sma = sum(data[i - n:i]) / n 
        temp_data[i] = sma
    return temp_data

def exponetial_moving_average(data, n):
    # Trước khi tính EMA, chỉ số SMA sẽ được tính trước

    # Sau đó nhân SMA với trọng số của EMA, được biết đến là "smoothing factor"
    # Với biểu thức được tính là: [2/(selected time period + 1)]
    temp_data = np.full(len(data), np.nan)
    EMA1 = simple_moving_average(data,n)[n]
    for i in range(n, len(temp_data)):
        s = 2 / (n + 1) #smoothing_factor
        EMA2 = data[i] * s + EMA1 * (1-s)
        temp_data[i] = EMA2
        EMA1 = EMA2
    return temp_data

def weighted_moving_average(data, n):
    temp_data = np.full(len(data), np.nan)
    sum_of_days = np.sum([i for i in range(1, n + 1)])
    weighting = np.array([i for i in range(1,n + 1)]) / sum_of_days
    for i in range(n, len(temp_data)):
        
        selected_data = data[i - n: i]
        WMA = np.sum(selected_data * weighting)
        temp_data[i] = WMA
    return temp_data

n_window = 30 # Số lượng ngày được ước lượng
real_data = df['Range (km)']

########################
# sma_data = simple_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], real_data, label='real_data')
# plt.plot(df['Azimuth (rad)'], sma_data, label = 'SMA')
# plt.title('Using Simple Moving Average to Smooth Data')
# plt.legend()
# plt.show()

# ########################
# ema_data = exponetial_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], real_data, label='real_data')
# plt.plot(df['Azimuth (rad)'], ema_data, label = 'EMA')
# plt.title('Using Exponential Moving Average to Smooth Data')
# plt.legend()
# plt.show()


########################
# ema_data = exponetial_moving_average(real_data, n_window)
# sma_data = simple_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], sma_data, label = 'SMA')
# plt.plot(df['Azimuth (rad)'], ema_data, label = 'EMA')
# plt.title('SMA V/S EMA')
# plt.legend()
# plt.show()

# ########################
# wma_data = weighted_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], real_data, label='real_data')
# plt.plot(df['Azimuth (rad)'], wma_data, label = 'EMA')
# plt.title('Using Exponential Moving Average to Smooth Data')
# plt.legend()
# plt.show()


# ########################
wma_data = weighted_moving_average(real_data, n_window)
sma_data = simple_moving_average(real_data, n_window)
plt.plot(df['Azimuth (rad)'], sma_data, label = 'SMA')
plt.plot(df['Azimuth (rad)'], wma_data, label = 'WMA')
plt.title('SMA V/S WMA')
plt.legend()
plt.show()