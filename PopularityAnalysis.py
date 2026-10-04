import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

actors_data = pd.read_csv('Datasets/credits_with_wiki.csv')
movies_data = pd.read_csv('Datasets/titles.csv')
merged_data = pd.merge(movies_data, actors_data, on='id', how='inner')

merged_data['is_famous'] = (merged_data['page_size'] > 0).astype(int)
merged_data['num_famous_actors'] = merged_data.groupby('id')['is_famous'].transform('sum')
X = merged_data[['num_famous_actors']]
Y = merged_data['imdb_votes'].fillna(merged_data['imdb_votes'].mean())

vote_bins = pd.qcut(Y, q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Unpopular', 'Mildly Popular', 'Popular', 'Very Popular', 'Extremely Popular'])

num_folds = 25
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

depth_values = range(2, 21)
param_grid = {'max_depth': depth_values}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=stratified_kfold, scoring='accuracy')
grid_search.fit(X, vote_bins)
best_classifier = grid_search.best_estimator_

classification_metrics = classification_report(vote_bins, best_classifier.predict(X), target_names=['Unpopular', 'Mildly Popular', 'Popular', 'Very Popular', 'Extremely Popular'], output_dict=True)
accuracy = accuracy_score(vote_bins, best_classifier.predict(X))
confusion_matrix_data = confusion_matrix(vote_bins, best_classifier.predict(X))

sns.heatmap(confusion_matrix_data, annot=True, fmt='', cmap='Blues', cbar=False, square=True, xticklabels=classification_metrics.keys(), yticklabels=classification_metrics.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print(f'Accuracy: {accuracy:.2f}')
for label, values in classification_metrics.items():
    if label != 'accuracy':
        print(f'{label.capitalize()}:')
        print(f'  Precision: {values["precision"]:.2f}')
        print(f'  Recall: {values["recall"]:.2f}')
        print(f'  F1-Score: {values["f1-score"]:.2f}')
