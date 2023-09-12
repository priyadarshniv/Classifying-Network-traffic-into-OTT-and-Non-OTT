import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset into a Pandas DataFrame (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv('modified_dataset.csv')

# Define the feature columns (excluding the 'label' column)
X = df.drop('label', axis=1)

# Define the target variable 'y' as the 'label' column
y = df['label']

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create an accuracy graph
train_sizes = [100, 500, 1000, 5000, 10000]  # Vary the training dataset size as needed
accuracies = []

for size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    model.fit(X_train_subset, y_train_subset)
    y_pred_subset = model.predict(X_test)
    accuracy_subset = accuracy_score(y_test, y_pred_subset)
    accuracies.append(accuracy_subset)

# Plot the accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, accuracies, marker='o')
plt.title('Accuracy vs. Training Size')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()