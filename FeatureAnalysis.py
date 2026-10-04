import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

titles_df = pd.read_csv('Datasets/titles.csv')
credits_df = pd.read_csv('Datasets/credits.csv')

merged_df = pd.merge(titles_df, credits_df, on='id', how='inner')

selected_features = ['release_year', 'runtime', 'seasons', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']

target_variable = 'imdb_score'

X = merged_df[selected_features]
y = merged_df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

if y_train.isnull().any():
    imputer = SimpleImputer(strategy='mean')
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1))
    
    y_train = y_train.ravel()

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model = RandomForestRegressor(random_state=0)
model.fit(X_train_imputed, y_train)

feature_importances = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances, align='center')
plt.yticks(range(len(feature_importances)), selected_features)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Analysis for IMDb Score Prediction (with Mean Imputation)')
plt.show()
