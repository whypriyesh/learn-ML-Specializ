import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# 1. Load Dataset
# ==========================
data = pd.read_csv("student_scores.csv")

X = data["Hours"].values
y = data["Scores"].values

# Convert to 2D for consistency
X = X.reshape(-1, 1)

# ==========================
# 2. Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 3. Initialize Parameters
# ==========================
w = 0.0
b = 0.0
alpha = 0.01
iterations = 1000
m = len(X_train)

# ==========================
# 4. Gradient Descent
# ==========================
for i in range(iterations):
    y_pred = w * X_train.flatten() + b

    dw = (1/m) * np.sum((y_pred - y_train) * X_train.flatten())
    db = (1/m) * np.sum(y_pred - y_train)

    w = w - alpha * dw
    b = b - alpha * db

# ==========================
# 5. Predictions
# ==========================
y_test_pred = w * X_test.flatten() + b

# ==========================
# 6. Evaluation
# ==========================
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Final Weight (w):", round(w, 3))
print("Final Bias (b):", round(b, 3))
print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))

# ==========================
# 7. Visualization
# ==========================
plt.scatter(X, y, label="Actual Data")
plt.plot(X, w * X.flatten() + b, color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.show()