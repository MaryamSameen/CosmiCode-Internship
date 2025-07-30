import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load dataset (Iris, binary classification for ROC-AUC)
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary: class 2 vs rest

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM (RBF kernel)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

metrics = {}

plt.figure(figsize=(8, 6))

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    metrics[name] = [acc, prec, rec, f1, roc_auc]
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Print metrics
print(f"{'Classifier':<20} {'Accuracy':<8} {'Precision':<9} {'Recall':<7} {'F1-score':<9} {'ROC-AUC':<8}")
for name, vals in metrics.items():
    print(f"{name:<20} {vals[0]:<8.2f} {vals[1]:<9.2f} {vals[2]:<7.2f} {vals[3]:<9.2f} {vals[4]:<8.2f}")