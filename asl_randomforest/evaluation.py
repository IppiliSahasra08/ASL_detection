from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

tree_counts = [10, 20, 50, 100, 150, 200]
accuracies = []

for n in tree_counts:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    accuracies.append(acc * 100)

plt.figure(figsize=(10, 6))
plt.plot(tree_counts, accuracies, marker='o', color='blue')
plt.title('Number of Trees vs Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy %')
plt.grid(True)
plt.savefig('/content/drive/MyDrive/trees_vs_accuracy.png')
plt.show()
print("Saved!")