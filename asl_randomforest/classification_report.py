from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

report = classification_report(y_test, y_pred, output_dict=True)
signs = sorted(set(y))
accuracies = [report[sign]['recall'] * 100 for sign in signs]

plt.figure(figsize=(12, 6))
bars = plt.bar(signs, accuracies, color='green')
plt.title('Accuracy per Sign')
plt.xlabel('Sign')
plt.ylabel('Accuracy %')
plt.ylim(0, 110)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.savefig('/content/drive/MyDrive/accuracy_per_sign.png')
plt.show()