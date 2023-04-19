import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
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

def exponential_moving_average(data, n):
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

# Function to find the best n_windows based on MSE coefficient 
def find_best_window(data ,data_real, f_type = 'sma'):
    temp_sma = simple_moving_average(data, 5)
    temp_sma = np.nan_to_num(temp_sma, 0)
    mae = mean_absolute_error(temp_sma, data_real)
    best_window = 1
    if f_type == 'wma':
        for j in range(1, 10):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            sma_data = weighted_moving_average(data, n_window)
            sma_data = np.nan_to_num(sma_data, 0)

            # Do dữ liệu khi tính trung bình sẽ mất n giá trị đầu, nên ta sẽ lật ngược tập dữ liệu để tính
            # Sau đó điền các giá trị bị thiếu bằng giá trị trung bình ngược

            # Tính giá trị trung bình phương vị ngược 1
            sma_data_revert = weighted_moving_average(np.flip(data, axis=0), n_window)
            sma_data_revert = np.nan_to_num(sma_data_revert, 0)
            sma_data[:n_window] = sma_data_revert[-n_window:] + sma_data[:n_window]

            # compute mae
            mae_data = mean_absolute_error(data_real, sma_data)
            if mae_data < mae:
                mae = mae_data
                best_window = j
                print(f'i = {j} and mae_data = {mae_data} ')
    elif f_type == 'ema':
        for j in range(1, 10):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            ema_data = exponential_moving_average(data, n_window)
            ema_data = np.nan_to_num(ema_data, 0)

            # Do dữ liệu khi tính trung bình sẽ mất n giá trị đầu, nên ta sẽ lật ngược tập dữ liệu để tính
            # Sau đó điền các giá trị bị thiếu bằng giá trị trung bình ngược

            # Tính giá trị trung bình phương vị ngược 1
            ema_data_revert = exponential_moving_average(np.flip(data, axis=0), n_window)
            ema_data_revert = np.nan_to_num(ema_data_revert, 0)
            ema_data[:n_window] = ema_data_revert[-n_window:] + ema_data[:n_window]

            # compute mae
            mae_data = mean_squared_error(data_real, ema_data)
            if mae_data < mae  :
                mae = mae_data
                best_window = j
            print(f'i = {j} and mae_data = {mae_data}')
    else:
        for j in range(1, 10):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            sma_data = simple_moving_average(data, n_window)
            sma_data = np.nan_to_num(sma_data, 0)

            # Do dữ liệu khi tính trung bình sẽ mất n giá trị đầu, nên ta sẽ lật ngược tập dữ liệu để tính
            # Sau đó điền các giá trị bị thiếu bằng giá trị trung bình ngược

            # Tính giá trị trung bình phương vị ngược 1
            sma_data_revert = simple_moving_average(np.flip(data, axis=0), n_window)
            sma_data_revert = np.nan_to_num(sma_data_revert, 0)
            sma_data[:n_window] = sma_data_revert[-n_window:] + sma_data[:n_window]

            # compute mae
            mae_data = mean_absolute_error(data_real, sma_data)
            if mae_data < mae :
                mae = mae_data
                best_window = j
            print(f'i = {j} and mae_data = {mae_data}')
    return best_window

########################
# sma_data = simple_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], real_data, label='real_data')
# plt.plot(df['Azimuth (rad)'], sma_data, label = 'SMA')
# plt.title('Using Simple Moving Average to Smooth Data')
# plt.legend()
# plt.show()

# ########################
# ema_data = exponential_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], real_data, label='real_data')
# plt.plot(df['Azimuth (rad)'], ema_data, label = 'EMA')
# plt.title('Using Exponential Moving Average to Smooth Data')
# plt.legend()
# plt.show()


########################
# ema_data = exponential_moving_average(real_data, n_window)
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
# wma_data = weighted_moving_average(real_data, n_window)
# sma_data = simple_moving_average(real_data, n_window)
# plt.plot(df['Azimuth (rad)'], sma_data, label = 'SMA')
# plt.plot(df['Azimuth (rad)'], wma_data, label = 'WMA')
# plt.title('SMA V/S WMA')
# plt.legend()
# plt.show()