# ------------------ Imports ------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Load Dataset ------------------
iris = load_iris()
X = iris.data
y = iris.target

split_values = np.linspace(0.001, 0.99, 1000)
acc_results = []

for split in split_values:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    acc_results.append(acc * 100)  # Optional: multiply by 100 for percentage

# ------------------ Predict Custom Sample ------------------
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print(f"\nPredicted Class for sample {sample}: {iris.target_names[prediction[0]]}")

# ------------------ Plot ------------------
plt.plot(split_values, acc_results, color='darkgreen')
plt.xlabel("Test Split Ratio")
plt.ylabel("Accuracy (%)")
plt.title("Naive Bayes Accuracy vs Test Split Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()
