import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Kỹ thuật này để khi mọi người dùng, không cần phải đổi tên đường dẫn 
dir_p = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(dir_p, '..', 'data', 'measData_ver2.csv')

# Đọc file
df = pd.read_csv(path_to_data)

# Hàm bổ trợ đánh dấu các vị trí trong đồ thị
def mark_position(x, y, position_name, color="r"):
    plt.plot(x, y, 'ro', markersize=10, color=color) 
    plt.text(x + 0.4, y + 0.4, position_name, fontsize=12,
            horizontalalignment='left', verticalalignment='bottom')


# Hàm vẽ quỹ đạo mục tiêu
def plot_orbit(df):
    # plot style
    sns.set_style('whitegrid')
    sns.set_context('talk')

    # Biểu diễn quỹ đạo
    fig = plt.figure(figsize=(10,8))  #  Chỉnh kích cỡ đồ thị khi hiển thị trong notebook
    plt.plot(df['x-axis'], df['y-axis'], '-', linewidth=1)

    # Đặt tiêu đề 
    plt.xlabel('Trục hoành (Độ lớn theo km)')
    plt.ylabel('Trục tung (Độ lớn theo km)')
    plt.title('Quỹ đạo của mục tiêu')
    
    
    # Thêm trục toạ độ Oxy
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)


    # Thêm nhãn điểm bắt đầu, điểm kết thúc và điểm "vị trí radar" trên đồ thị
    mark_position(df['x-axis'][0], df['y-axis'][0], "t = 0s")  #Điểm bắt đầu
    mark_position(df['x-axis'][len(df) - 1], df['y-axis'][len(df) - 1], "t = 4990s")  #Điểm kết thúc

    mark_position(0, 0, "Vị trí radar", color="g")  # Vị trí radar
    

    # Đánh dấu thêm một vài thời điểm:
    for i in range(1, 5):
        mark_position(df['x-axis'][i * 100], df['y-axis'][i * 100], f"t = {i * 1000}s")  #Điểm kết thúc


    plt.show()


