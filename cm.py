from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

actual =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]


cm = confusion_matrix(actual, predicted)
print("CONFUSION MATRIX:\n", cm)

TN, TP, FN, FP = cm.ravel()

print(f"\nTrue Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")


accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1_score  = 2 * (precision * recall) / (precision + recall)

print("\n---- METRICS ----")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1_score:.2f}")


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap="Oranges")
plt.title("Confusion Matrix with Metrics")
plt.show()