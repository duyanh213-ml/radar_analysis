from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from moving_average import simple_moving_average

df = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\measData_ver2.csv')
df_real = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\groundtruthData_ver2.csv')

# Đọc dữ liệu đầu vào và chuẩn hóa
data_raw = df['x-axis'].values
data_real = df_real['x-axis'].values

scaler = StandardScaler()
norm_raw = scaler.fit_transform(data_raw.reshape(-1,1)).reshape(1,-1)
norm_real = scaler.fit_transform(data_real.reshape(-1,1)).reshape(1,-1)

def cost_function(data_raw, data_real, window_size):
    window_size = int(round(window_size))
    sma = simple_moving_average(data_raw, window_size)[window_size:]
    actual = data_real[window_size:]
    return mean_squared_error(actual, sma)

def gradient(data_raw, data_real, window_size):
    eps = 1e-4
    cost = cost_function(data_raw, data_real, window_size)
    grad = 0
    for i in range(int(window_size)):
        h = eps
        grad += (cost_function(data_raw, data_real, int(window_size + h)) - cost_function(data_raw, data_real, int(window_size - h))) / (2 * eps)
    return grad

def adam_optimizer(data_raw, data_real, initial_window_size, alpha=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000):
    # Initialize variables
    m = 0
    v = 0
    t = 0
    window_size = initial_window_size

    for i in range(num_iterations):
        grad = gradient(data_raw, data_real, window_size)
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        window_size -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return window_size

initial_window_size = 10
optimal_window_size = adam_optimizer(data_raw, data_real, initial_window_size)
ma = simple_moving_average(data_raw, round(optimal_window_size))

print("optimal_window_size: ", optimal_window_size)
print('rmse: ', np.sqrt(mean_squared_error(ma[round(optimal_window_size):], data_real[round(optimal_window_size):])))
