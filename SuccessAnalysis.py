import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
credits_df = pd.read_csv('Datasets/credits_with_wiki.csv')
titles_df = pd.read_csv('Datasets/titles.csv')

merged_df = pd.merge(titles_df, credits_df, on='id', how='inner')

merged_df['isFamous'] = (merged_df['page_size'] > 0).astype(int)
merged_df['numFamousActors'] = merged_df.groupby('id')['isFamous'].transform('sum')

X = merged_df[['numFamousActors']]
Y = merged_df['imdb_score'].fillna(merged_df['imdb_score'].mean())  # Fill missing values with the mean
skewness = Y.skew()

custom_bins = pd.qcut(Y, q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Flop', 'Unsuccessful', 'Mildly Successful', 'Successful', 'Very Successful'])

num_folds = 25
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, None]

param_grid = {'max_depth': depths}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=skf, scoring='accuracy')
grid_search.fit(X, custom_bins)

best_model = grid_search.best_estimator_

precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []  # To store accuracy scores

confusion_matrices = []

for i, (train_idx, test_idx) in enumerate(skf.split(X, custom_bins)):
    X_train, y_train = X.iloc[train_idx], custom_bins.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], custom_bins.iloc[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    precision_scores.append(classification_report_dict['weighted avg']['precision'])
    recall_scores.append(classification_report_dict['weighted avg']['recall'])
    f1_scores.append(classification_report_dict['weighted avg']['f1-score'])

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_accuracy = np.mean(accuracy_scores)  # Calculate the average accuracy

average_confusion_matrix = np.mean(confusion_matrices, axis=0).astype(int)

class_labels = ['Flop', 'Unsuccessful', 'Mildly Successful', 'Successful', 'Very Successful']

labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [average_accuracy, average_precision, average_recall, average_f1]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.bar(labels, scores, color=['purple', 'blue', 'green', 'orange'])
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Average Metrics')

plt.subplot(1, 2, 2)
sns.heatmap(average_confusion_matrix, annot=True, fmt='', cmap='Blues', cbar=False, square=True, xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Average Confusion Matrix')

plt.tight_layout()
plt.show()
