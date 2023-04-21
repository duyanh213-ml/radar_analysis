import numpy as np


def kalman_filter(measurements, delta_t, acceleration_noisy_var, R, x_init, P_init):
    #  Ma trận chuyển trạng thái (state transition matrix) 
    F = np.array([
        [1, delta_t, 0.5 * delta_t ** 2, 0, 0, 0],
        [0, 1, delta_t, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, delta_t, 0.5 * delta_t ** 2],
        [0, 0, 0, 0, 1, delta_t],
        [0, 0, 0, 0, 0, 1]
    ])

    # Ma trận nhiễu gây bởi gia tốc (process noise matrix)
    Q = acceleration_noisy_var * np.array([
        [0.25 * delta_t ** 4, 0.5 * delta_t ** 3, 0.5 * delta_t ** 2, 0, 0, 0],
        [0.5 * delta_t ** 3, delta_t ** 2, delta_t, 0, 0, 0],
        [0.5 * delta_t ** 2, delta_t, 1, 0, 0, 0],
        [0, 0, 0, 0.25 * delta_t ** 4, 0.5 * delta_t ** 3, 0.5 * delta_t ** 2],
        [0, 0, 0, 0.5 * delta_t ** 3, delta_t ** 2, delta_t],
        [0, 0, 0, 0.5 * delta_t ** 2, delta_t, 1]
    ])

    # Ma trận quán sát (Observation matrix)
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    # Ma trận đơn vị
    I = np.identity(len(x_init))

    # List chứa các ước lượng
    X = [x_init]
    # List chứa hiệp phương sai các ước lượng
    P = [P_init]
    # List lưu lại các ma trận trọng số Kalman gain
    Kalman = []

    # Thực hiện thuật toán Kalman filter
    for i in range(measurements.shape[0]):

        # 5 phương trình của Kalman filter:

        # State extrapolation
        x_extra = F @ X[i]

        # Estimation Covariance extrapolation
        P_extra = F @ P[i] @ F.T + Q

        # Kalman gain
        K = P_extra @ H.T @ np.linalg.inv((H @ P_extra @ H.T + R))
        
        z = measurements[i]

        # State update
        x_update = x_extra + K @ (z - H @ x_extra)

        # Estimation Covariance update
        P_update = (I - K @ H) @ P_extra @ (I - K @ H).T + K @ R @ K.T
        
        X.append(x_update)
        P.append(P_update)
        Kalman.append(K)

    X = np.array(X)
    P = np.array(P)
    K = np.array(K)

    return_data = {
        "Estimations" : X,
        "Estimation Uncertainty": P,
        "Kalman gain": K
    }

    return return_data
    

