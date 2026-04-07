import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Authentic", "Forged"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png", bbox_inches="tight")
print("confusion_matrix.png created successfully")