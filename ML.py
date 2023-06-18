import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load your dataset and split it into training and testing sets
# X is the feature matrix, y is the target variable
df = pd.read_csv('your_dataset.csv')
X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1_score = f1_score(y_test, dt_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
dt_specificity = dt_confusion_matrix[0, 0] / (dt_confusion_matrix[0, 0] + dt_confusion_matrix[0, 1])
dt_sensitivity = dt_recall

print("Decision Tree Metrics:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1_score)
print("Specificity:", dt_specificity)
print("Sensitivity:", dt_sensitivity)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1_score = f1_score(y_test, nb_predictions)
nb_confusion_matrix = confusion_matrix(y_test, nb_predictions)
nb_specificity = nb_confusion_matrix[0, 0] / (nb_confusion_matrix[0, 0] + nb_confusion_matrix[0, 1])
nb_sensitivity = nb_recall

print("Naive Bayes Metrics:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1_score)
print("Specificity:", nb_specificity)
print("Sensitivity:", nb_sensitivity)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1_score = f1_score(y_test, svm_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
svm_specificity = svm_confusion_matrix[0, 0] / (svm_confusion_matrix[0, 0] + svm_confusion_matrix[0, 1])
svm_sensitivity = svm_recall

print("SVM Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1_score)
print("Specificity:", svm_specificity)
print("Sensitivity:", svm_sensitivity)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
knn_specificity = knn_confusion_matrix[0, 0] / (knn_confusion_matrix[0, 0] + knn_confusion_matrix[0, 1])
knn_sensitivity = knn_recall

print("KNN Metrics:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1_score)
print("Specificity:", knn_specificity)
print("Sensitivity:", knn_sensitivity)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_specificity = rf_confusion_matrix[0, 0] / (rf_confusion_matrix[0, 0] + rf_confusion_matrix[0, 1])
rf_sensitivity = rf_recall

print("Random Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1_score)
print("Specificity:", rf_specificity)
print("Sensitivity:", rf_sensitivity)
