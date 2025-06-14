import pandas as pd
import numpy as np

def gradient_descent(x, y, lr=0.1, epochs=3000):
    # scale x and y using min-max scaling
    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    # initiate parameters
    b = 0.0  # intercept
    m = 0.0  # slope
    n = len(y_scaled)

    # perform gradient descent
    for epoch in range(epochs):
        y_pred = b + m * x_scaled 
        error = y_scaled - y_pred 
        cost = np.mean(error ** 2)

        # calcualte gradient 
        db = -2 * np.mean(error)
        dm = -2 * np.mean(error * x_scaled)

        # update parameters 
        b -= lr * db 
        m -= lr * dm 

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")  
    
    # Scale back the coefficients to original scale
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    b, m = gradient_descent(x, y)

    print(f"Final Results: m={round(m, 0)}, b={round(b, 0)}")