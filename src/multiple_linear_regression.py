"""
Multiple Linear Regression - From Scratch
This script demonstrates how a computer learns to predict a value (like house price)
based on multiple features (like size and number of bedrooms).
"""

import numpy as np
import copy

def get_dummy_data():
    """
    Creates a synthetic dataset of 5 houses.
    Features (X): [Size in sq ft, Number of bedrooms]
    Target (y): Price in $1000s
    """
    X_train = np.array([
        [2104, 5],
        [1416, 3],
        [1534, 3],
        [852,  2],
        [1940, 4]
    ])
    
    y_train = np.array([460, 232, 315, 178, 240])
    return X_train, y_train

def predict(x, w, b):
    """
    Predictions are made using the formula: y_hat = (x dot w) + b
    x: Features of a single house
    w: Model weights (importance of each feature)
    b: Model bias (base price)
    """
    # np.dot pairs each feature with its corresponding weight and sums them!
    return np.dot(x, w) + b

def compute_cost(X, y, w, b):
    """
    Calculates the Mean Squared Error (MSE).
    This tells us how "wrong" our model currently is. Lower cost = better model.
    We square the error to ensure negative and positive errors don't cancel out,
    and to heavily penalize really bad predictions.
    """
    m = X.shape[0] # Number of houses (5)
    total_cost = 0.0
    
    for i in range(m):
        y_hat = predict(X[i], w, b)
        error = y_hat - y[i]
        total_cost += (error ** 2)
        
    return total_cost / (2 * m)

def compute_gradient(X, y, w, b):
    """
    Calculates the "slope" (gradient) of our cost function.
    This tells the model which direction to adjust the weights and bias 
    to make the Cost (error) go down.
    """
    m, n = X.shape # m = houses, n = features
    
    # Store the gradients (adjustments) for weights and bias
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    
    for i in range(m):
        error = predict(X[i], w, b) - y[i]
        dj_db += error
        
        for j in range(n):
            dj_dw[j] += error * X[i, j]
            
    # Average the gradients across all houses
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    The training loop. It repeatedly calculates the gradient and updates 
    the parameters (w, b) to minimize the cost.
    alpha: Learning rate (how big of a step we take down the slope)
    """
    w = copy.deepcopy(w_init)
    b = b_init
    
    for i in range(num_iters):
        # 1. Find the direction to move
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # 2. Update the weights and bias by taking a small step (alpha)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        
        # 3. Print progress evenly across the iterations
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:5d}: Cost {cost:.2f}")
            
    return w, b

if __name__ == "__main__":
    # --- 1. Load Data ---
    print("--- Multiple Linear Regression Demo ---")
    X_train, y_train = get_dummy_data()
    print(f"Dataset Loaded: {X_train.shape[0]} Houses, {X_train.shape[1]} Features")
    
    # --- 2. Initialize Parameters ---
    # We have 2 features, so we need 2 weights. Start both at 0.0
    initial_w = np.zeros(X_train.shape[1]) 
    initial_b = 0.0
    
    # --- 3. Training Settings ---
    # We use a tiny learning rate because our features are not scaled
    learning_rate = 5.0e-7 
    iterations = 1000
    
    # --- 4. Train the Model! ---
    print("\nStarting Training (Gradient Descent)...")
    final_w, final_b = gradient_descent(X_train, y_train, initial_w, initial_b, learning_rate, iterations)
    
    print("\n--- Training Complete ---")
    print(f"Learned weights (w): {final_w}")
    print(f"Learned bias (b):    {final_b}")
    
    # --- 5. Test the Model ---
    first_house = X_train[0]
    actual_price = y_train[0]
    predicted_price = predict(first_house, final_w, final_b)
    
    print("\n--- Final Prediction Test ---")
    print(f"House Size: {first_house[0]} sq ft | Bedrooms: {first_house[1]}")
    print(f"Model Predicts: ${predicted_price:.2f}k")
    print(f"Actual Price:   ${actual_price}k")
    print("---------------------------------------")
