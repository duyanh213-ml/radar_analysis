import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# Kỹ thuật này để khi mọi người dùng, không cần phải đổi tên đường dẫn 
dir_p = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(dir_p, '..', 'data', 'measData_ver2.csv')
path_to_real = os.path.join(dir_p, '..', 'data', 'groundtruthData_ver2.csv')
# Đọc file
df = pd.read_csv(path_to_data)
df_real = pd.read_csv(path_to_real)
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
def compute_euclid(x_0,y_0,x_1,y_1):
    x0 = np.nan_to_num(x_0, 0)
    x1 = np.nan_to_num(x_1, 0)
    y0 = np.nan_to_num(y_0, 0)
    y1 = np.nan_to_num(y_1, 0)
    
    distance = np.sqrt((x0 - x1) ** 2+ (y0 - y1) **2)
     
    return distance

# Function to find the best n_windows based on RMSE coefficient 
def find_best_window(data ,data_real, f_type = 'sma'):
    # Separate x, y in data
    x_0 = data[0]
    x_1 = data_real[0]
    y_0 = data[1]
    y_1= data_real[1]
    
    # Caculate distance from raw data to real data
    distance_0 = compute_euclid(x_0, y_0, x_1, y_1)
    
    # Caculate distance from filter data to real data
    x_filter = simple_moving_average(x_0, 5)
    y_filter = simple_moving_average(y_0, 5)
    distance_1 = compute_euclid(x_filter, y_filter, x_1, y_1)
    
    # Compute RMSE
    mse = mean_squared_error(distance_0, distance_1)
    rmse = np.sqrt(mse)
    best_window = 1
    
    arr_rmse = []
    if f_type == 'wma':
        for j in range(1, 30):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            wma_x = weighted_moving_average(x_0, n_window)
            wma_y =  weighted_moving_average(y_0, n_window)

            # Do dữ liệu khi tính trung bình sẽ mất n giá trị đầu, nên ta sẽ lật ngược tập dữ liệu để tính
            # Sau đó điền các giá trị bị thiếu bằng giá trị trung bình ngược

            # Tính giá trị trung bình phương vị ngược 1
            wma_x_revert = weighted_moving_average(np.flip(x_0, axis=0), n_window)
            wma_y_revert = weighted_moving_average(np.flip(y_0, axis=0), n_window)
            
            wma_x[:n_window] = wma_x_revert[-n_window:] + wma_x[:n_window]
            wma_y[:n_window] = wma_y_revert[-n_window:] + wma_y[:n_window]
            # compute mae
            distance_1 = compute_euclid(wma_x, wma_y, x_1, y_1)
            mse_1 = mean_squared_error(distance_1, distance_0)
            rmse_1 = np.sqrt(mse_1)
            arr_rmse.append(rmse_1)
            if rmse_1 > rmse:
                rmse = rmse_1
                best_window = j
                print(f'i = {j} and rmse_data = {rmse_1} ')
    elif f_type == 'ema':
        for j in range(1, 30):
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
            arr_rmse.append(mae_data)
            if mae_data < mae  :
                mae = mae_data
                best_window = j
            print(f'i = {j} and mae_data = {mae_data}')
    else:
        for j in range(1, 30):
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
            arr_rmse.append(mae_data)
            if mae_data < mae :
                mae = mae_data
                best_window = j
            print(f'i = {j} and mae_data = {mae_data}')
    return (best_window, arr_rmse)

scaler = StandardScaler()

raw_x = scaler.fit_transform(df['x-axis'].to_numpy().reshape(-1,1))
real_x = scaler.fit_transform(df_real['x-axis'].to_numpy().reshape(-1,1))
raw_y = scaler.fit_transform(df['y-axis'].to_numpy().reshape(-1,1))
real_y = scaler.fit_transform(df['y-axis'].to_numpy().reshape(-1,1))

# n_window = find_best_window([raw_x, raw_y], [real_x, real_y], f_type='wma')
n_window = 10
# print(n_window)

wma_x = weighted_moving_average(raw_x, n_window)
wma_x = np.nan_to_num(wma_x, 0)
wma_y =  weighted_moving_average(raw_y, n_window)
wma_y = np.nan_to_num(wma_y, 0)

# Do dữ liệu khi tính trung bình sẽ mất n giá trị đầu, nên ta sẽ lật ngược tập dữ liệu để tính
# Sau đó điền các giá trị bị thiếu bằng giá trị trung bình ngược

# Tính giá trị trung bình phương vị ngược 1
wma_x_revert = weighted_moving_average(np.flip(raw_x, axis=1), n_window)
wma_x_revert = np.nan_to_num(wma_x_revert, 0)
wma_y_revert = weighted_moving_average(np.flip(raw_y, axis=1), n_window)
wma_y_revert = np.nan_to_num(wma_y_revert, 0)
wma_x[:n_window] = wma_x_revert[-n_window:] + wma_x[:n_window]
wma_y[:n_window] = wma_y_revert[-n_window:] + wma_y[:n_window]
wma_y = wma_y.reshape(-1, 1)
wma_x = wma_x.reshape(-1,1)
d0 = np.array(compute_euclid(raw_x, raw_y, real_x, real_y))
d1 = np.array(compute_euclid(wma_x, wma_y, real_x, real_y))

rmse = np.sqrt(mean_squared_error(d0, d1))
print(rmse)


# print(d0)
# print(d1)
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