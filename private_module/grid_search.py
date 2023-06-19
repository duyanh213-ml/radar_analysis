import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from private_module.moving_average import exponential_moving_average, simple_moving_average, weighted_moving_average
# from moving_average import exponential_moving_average, simple_moving_average, weighted_moving_average

df = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\measData_ver2.csv')
df_real = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\groundtruthData_ver2.csv')

# Cost function
def mde(x_true, x_pred,y_true, y_pred):
    return np.mean(np.sqrt((x_pred - x_true) **2 + (y_pred - y_true) ** 2))

def find_best_window(x_raw, y_raw, x_real, y_real, f_type='ema'):
    if f_type == 'ema':
        list_index = []
        # Create a list contains a list of value want to compare
        list_smooth = np.linspace(5,6,200)
        for i in list_smooth: # values of smooth
            for j in range(1, 30): #size of window
                ma_x = exponential_moving_average(x_raw, j, smooth=i)
                ma_y = exponential_moving_average(y_raw, j, smooth=i)
                mde_0 = mde(x_real, ma_x, y_real, ma_y)
                list_index.append([i,j,mde_0])
                print('smooth:', i, 'window:', j, 'mde:', mde_0)
        # Find the best RMSE in this list
            print('smooth:', i)
        best_tuple = min(list_index, key=lambda x: x[2])
        # return best_tuple
        return best_tuple
    elif f_type == 'wma':
        list_index = []
        # Create a list contains a list of value want to compare
        for j in range(1, 30): #size of windowe
            ma_x = weighted_moving_average(x_raw, j)
            ma_y = weighted_moving_average(y_raw, j)
            mde_0 = mde(x_real[j:], ma_x[j:], y_real[j:], ma_y[j:])
            list_index.append([j,mde_0])
            print('window:', j)
            print(mde_0)
    # Find the best RMSE in this list
        best_tuple = min(list_index, key=lambda x: x[1])
        return best_tuple
    else:
        list_index = []
        # Create a list contains a list of value want to compare
        for j in range(1, 30): #size of windowe
            ma_x = simple_moving_average(x_raw, j)
            ma_y = simple_moving_average(y_raw, j)
            mde_0 = mde(x_real[j:], ma_x[j:], y_real[j:], ma_y[j:])
            list_index.append([j,mde_0])
            print('window:', j)
            print(mde_0)
    # Find the best RMSE in this list
        best_tuple = min(list_index, key=lambda x: x[1])
        return best_tuple
# Function to find the best n_windows based on RMSE coefficient 
def find_best_method(data ,data_real, f_type = 'sma', num_loop=100, smooth=2):
    # Separate x, y in data
    x_raw = data[0]
    x_real = data_real[0]
    y_raw = data[1]
    y_real = data_real[1]

    arr_rmse = []
    if f_type == 'wma':
        for j in range(1, num_loop):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            ma_x = weighted_moving_average(x_raw, n_window)
            ma_y =  weighted_moving_average(y_raw, n_window)

            # compute mde
            mde_0 = mde(x_real[n_window:], ma_x[n_window:], y_real[n_window:], ma_y[n_window:])
            arr_rmse.append(mde_0)
    elif f_type == 'ema':
        for j in range(1, num_loop):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            ma_x = exponential_moving_average(x_raw, n_window,smooth=smooth)
            ma_y =  exponential_moving_average(y_raw, n_window,smooth=smooth)

            # compute mde
            mde_0 = mde(x_real, ma_x, y_real, ma_y)
            arr_rmse.append(mde_0)
    else:
        for j in range(1, num_loop):
            n_window = j
            # Ước lượng trung bình dữ liệu 1
            ma_x = simple_moving_average(x_raw, n_window)
            ma_y =  simple_moving_average(y_raw, n_window)

            # compute mde
            mde_0 = mde(x_real[n_window:], ma_x[n_window:], y_real[n_window:], ma_y[n_window:])
            arr_rmse.append(mde_0)
    return arr_rmse

# x_raw = df['x-axis'].values
# x_real = df_real['x-axis'].values
# y_raw = df['y-axis'].values
# y_real = df_real['y-axis'].values

# # temp = find_best_method([x_raw, y_raw], [x_real, y_real], f_type='ema')
# temp = find_best_window(x_raw, y_raw, x_real, y_real)
# temp_df = pd.DataFrame(temp, columns=['Smooth', 'Window', 'MDE'])

# # filter the DataFrame to only include rows where MDE is less than or equal to 1
# df_filtered = temp_df[temp_df['MDE'] <= 1]

# # plot the filtered DataFrame
# df_filtered.plot(x='Smooth', y='MDE', kind='line')
# plt.show()

# best_tuple = min(temp, key=lambda x: x[2])
# print(best_tuple)
# # # print(mde(x_real, x_raw, y_real, y_raw))