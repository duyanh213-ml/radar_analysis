import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import itertools
from scipy.optimize import minimize
# Kỹ thuật này để khi mọi người dùng, không cần phải đổi tên đường dẫn 
dir_p = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(dir_p, '..', 'data', 'measData_ver2.csv')
path_to_real = os.path.join(dir_p, '..', 'data', 'groundtruthData_ver2.csv')
# Đọc file
df = pd.read_csv(path_to_data)
df_real = pd.read_csv(path_to_real)

def simple_moving_average(data, window_size):
    n = int(round(window_size))
    temp_data = np.full(len(data), np.nan)
    for i in range(n, len(data)):
        sma = sum(data[i - n:i]) / n 
        temp_data[i] = sma
    return temp_data

def exponential_moving_average(data, n, smooth=2):
    # Trước khi tính EMA, chỉ số SMA sẽ được tính trước

    # Sau đó nhân SMA với trọng số của EMA, được biết đến là "smoothing factor"
    # Với biểu thức được tính là: [2/(selected time period + 1)]
    window_size = n
    alpha = smooth / (window_size + 1)
    y_pred = np.zeros_like(data)
    y_pred[0] = data[0]
    for t in range(1, len(data)):
        y_pred[t] = alpha * data[t] + (1 - alpha) * y_pred[t-1]
    return y_pred

def weighted_moving_average(data, n):
    temp_data = np.full(len(data), np.nan)

    sum_of_days = np.sum([i for i in range(1, n + 1)])
    weighting = np.array([i for i in range(1,n + 1)]) / sum_of_days
    for i in range(n, len(temp_data)):
        
        selected_data = data[i - n: i]
        WMA = np.sum(selected_data * weighting)
        temp_data[i] = WMA
    return temp_data


data_real = df_real.iloc[:, [-2, -1]].values
data_raw = df.iloc[:, [-2, -1]].values