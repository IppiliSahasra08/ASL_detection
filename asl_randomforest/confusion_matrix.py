import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=sorted(set(y)),
            yticklabels=sorted(set(y)))
plt.title('Confusion Matrix')
plt.ylabel('Actual Sign')
plt.xlabel('Predicted Sign')
plt.savefig('/content/drive/MyDrive/confusion_matrix.png')
plt.show()
print("Saved!")