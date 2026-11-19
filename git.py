import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("student_marks_hours.csv")

# Features and target
X = df[["Hours_Studied"]]
y = df["Marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict marks for 7.5 hours of study
new_prediction = model.predict(pd.DataFrame({"Hours_Studied": [7.5]}))
print("Predicted Marks:", new_prediction[0])