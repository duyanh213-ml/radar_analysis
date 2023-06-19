import pygame
import numpy as np
import numpy.random as random
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1400, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Mouse Tracking with Kalman Filter')

# Define colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up clock for frame rate
clock = pygame.time.Clock()

# Define Kalman filter parameters
dt = 1  # Time step in seconds
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * 1
R = np.array([[1, 0], [0, 1]]) * 16
x = np.array([[0, 0, 0, 0]]).T
P = np.eye(4) * 1000

# Initialize mouse state
mouse_down = False
mouse_pos = np.zeros((2, 1))

# Store truth positions and filtered mouse positions
truth_positions = []
filtered_positions = []

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False

    # Update mouse position with noise and store as truth
    if mouse_down:
        noise = random.multivariate_normal([0, 0], R).reshape(2, 1)
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float).reshape(2, 1) + noise
        truth_positions.append(mouse_pos.flatten())

    # Prediction step
    x = A @ x
    P = A @ P @ A.T + Q

    # Update step
    if mouse_down:
        z = mouse_pos - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ z
        P = (np.eye(4) - K @ H) @ P

    # Append filtered mouse position to list
    filtered_positions.append(x[:2])

    # Clear screen
    screen.fill(BLACK)

    # Draw truth positions
    for pos in truth_positions:
        pos_int = pos.astype(int)  # Convert to integer array
        pygame.draw.circle(screen, RED, tuple(pos_int), 1)

    # Draw filtered positions
    for pos in filtered_positions:
        pos_int = pos.astype(int).flatten()  # Convert to integer array and flatten to (x, y)
        pygame.draw.circle(screen, GREEN, tuple(pos_int), 1)

    if mouse_down:
        pygame.draw.circle(screen, RED, tuple(mouse_pos.flatten().astype(int)), 1)

    # Draw Kalman line
    for i in range(len(filtered_positions) - 1):
        pos1 = filtered_positions[i].astype(int).flatten()
        pos2 = filtered_positions[i + 1].astype(int).flatten()
        pygame.draw.line(screen, GREEN, tuple(pos1), tuple(pos2), 1)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

