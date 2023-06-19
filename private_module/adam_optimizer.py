import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import pandas as pd
df = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\measData_ver2.csv')
df_real = pd.read_csv('D:\\dulieuD\\Program Language\\New folder\\radar\\radar_analysis\\data\\groundtruthData_ver2.csv')

def mde(x_true, x_pred,y_true, y_pred):

    return np.mean(np.sqrt(((x_pred - x_true) **2 + (y_pred - y_true) ** 2)))
def ema(x, window_size, smooth=2):
    alpha = smooth / (window_size + 1)
    y_pred = np.zeros_like(x)
    y_pred[0] = x[0]
    for t in range(1, len(x)):
        y_pred[t] = alpha * x[t] + (1 - alpha) * y_pred[t-1]
    return y_pred
def adam_optimizer(x_raw,y_raw, x_real, y_real, window_size_init, smooth_factor_init, learning_rate=0.01,beta1=0.9, beta2=0.999, epsilon=1e-8, max_iteration=500):
    window_size = window_size_init
    smooth_factor = smooth_factor_init

    m_w = 0
    v_w = 0
    m_s = 0
    v_s = 0
    history = []
    for t in range(1, max_iteration + 1):
        grad_w = np.zeros_like(window_size)
        grad_s = np.zeros_like(smooth_factor)
        for i in range(len(x_raw)):
            delta_w = 20
            delta_s = 0.5
            x_predict_plus = ema(x_raw + delta_w, window_size, smooth=smooth_factor)
            x_predict_minus = ema(x_raw - delta_w, window_size, smooth=smooth_factor)
            y_predict_plus = ema(y_raw + delta_w, window_size, smooth=smooth_factor)
            y_predict_minus = ema(y_raw - delta_w, window_size, smooth=smooth_factor)

            grad_w = (mde(x_real, x_predict_plus, y_real, y_predict_plus) - mde(x_real, x_predict_minus, y_real, y_predict_minus)) / (2*delta_w)

            x_predict_plus = ema(x_raw, window_size, smooth=smooth_factor +delta_s)
            x_predict_minus = ema(x_raw, window_size, smooth=smooth_factor - delta_s)
            y_predict_plus = ema(y_raw, window_size, smooth=smooth_factor + delta_s)
            y_predict_minus = ema(y_raw, window_size, smooth=smooth_factor - delta_s)
            grad_s = (mde(x_real, x_predict_plus, y_real, y_predict_plus) - mde(x_real, x_predict_minus, y_real, y_predict_minus)) / (2*delta_s)
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w**2
        m_s = beta1 * m_s + (1 - beta1) * grad_s
        v_s = beta2 * v_s + (1 - beta2) * grad_s**2
        m_w_hat = m_w / (1 - beta1**t)
        v_w_hat = v_w / (1 - beta2**t)
        m_s_hat = m_s / (1 - beta1**t)
        v_s_hat = v_s / (1 - beta2**t)
        window_size -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        smooth_factor -= learning_rate * m_s_hat / (np.sqrt(v_s_hat) + epsilon)
        smooth_factor = round(smooth_factor, 4)
        ema_x = ema(x_raw, int(round(window_size)), smooth = smooth_factor)
        ema_y = ema(y_raw, int(round(window_size)), smooth = smooth_factor)
        loss_value = mde(x_real, ema_x, y_real, ema_y)
        print(f'epoch {t}: w: {window_size}, s:{smooth_factor}')
        print(f'Loss: {loss_value}')
        history.append([window_size, smooth_factor, loss_value])
    return window_size, smooth_factor, history

x_raw = df['x-axis'].values
x_real = df_real['x-axis'].values
y_raw = df['y-axis'].values
y_real = df_real['y-axis'].values
window_size_init = 19
smooth_factor_init = 16


optimizer = adam_optimizer(x_raw, y_raw,x_real, y_real,window_size_init, smooth_factor_init)
history = optimizer[2]
np.savetxt('history_ema_optimizer_6.csv',history, delimiter=',')
print(optimizer[0:2])

# ema_x = ema(x_raw, window_size_init,smooth=smooth_factor_init)
# ema_y = ema(y_raw, window_size_init,smooth=smooth_factor_init)
# loss_value = mde(x_real, ema_x, y_real, ema_y)
# print(loss_value)
# print(mean_squared_error(ema_x, x_real))
# temp = x_raw + 5
# print(temp)
