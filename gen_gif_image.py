import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import os
from private_module.moving_average import simple_moving_average
# Khai báo đường dẫn đến dữ liệu
dir_p = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(dir_p, 'data', 'measData_ver2.csv')
path_to_real = os.path.join(dir_p,'data', 'groundtruthData_ver2.csv')
# Đoạn code sau để lấy dữ liệu từ file
df = pd.read_csv(path_to_data)
df_real = pd.read_csv(path_to_real)
# Đoạn code tạo ra các hình ảnh cho mỗi frame của quỹ đạo mục tiêu
N = 100
x_raw = df['x-axis'].values
y_raw = df['y-axis'].values
x_real = df_real['x-axis'].values
y_real = df_real['y-axis'].values

fig, ax = plt.subplots()
images = []

for i in range(1,N+1):
    x_ma = simple_moving_average(x_raw, i )   
    y_ma = simple_moving_average(y_raw, i)
    ax.clear()  # clear the previous plot
    ax.plot(x_raw[i:], y_raw[i:], label='RAW')
    ax.plot(x_ma[i:], y_ma[i:], label='SMA')
    ax.plot(x_real[i:], y_real[i:], label='REAL')
    ax.set_title('Window = {}'.format((i)))
    ax.legend()
    fig.canvas.draw()

    # Chuyển đổi nội dung của figure sang dạng array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(img)

# Đoạn code lưu các ảnh thành một file GIF
imageio.mimsave('display_sma.gif', images, fps=5)
