import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/preprocessed_data.csv')

X_description = df[['sentiment_score_description']]
X_title = df[['sentiment_score_titles']]
y = df['imdb_score']

X_description_train, X_description_test, y_train, y_test = train_test_split(X_description, y, test_size=0.2, random_state=42)
X_title_train, X_title_test, y_train, y_test = train_test_split(X_title, y, test_size=0.2, random_state=42)

sentiment_model_description = RandomForestRegressor(random_state=42)
sentiment_model_title = RandomForestRegressor(random_state=42)

sentiment_model_description.fit(X_description_train, y_train)
sentiment_model_title.fit(X_title_train, y_train)

y_pred_holdout_description = sentiment_model_description.predict(X_description_test)
y_pred_holdout_title = sentiment_model_title.predict(X_title_test)

mse_holdout_description = mean_squared_error(y_test, y_pred_holdout_description)
r2_holdout_description = r2_score(y_test, y_pred_holdout_description)

mse_holdout_title = mean_squared_error(y_test, y_pred_holdout_title)
r2_holdout_title = r2_score(y_test, y_pred_holdout_title)

k = 10
mse_scores_description = -cross_val_score(sentiment_model_description, X_description, y, scoring='neg_mean_squared_error', cv=k)
r2_scores_description = cross_val_score(sentiment_model_description, X_description, y, scoring='r2', cv=k)

mse_scores_title = -cross_val_score(sentiment_model_title, X_title, y, scoring='neg_mean_squared_error', cv=k)
r2_scores_title = cross_val_score(sentiment_model_title, X_title, y, scoring='r2', cv=k)

mean_mse_cv_description = mse_scores_description.mean()
mean_r2_cv_description = r2_scores_description.mean()

mean_mse_cv_title = mse_scores_title.mean()
mean_r2_cv_title = r2_scores_title.mean()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_holdout_description, alpha=0.5)
plt.xlabel('Actual IMDb Scores')
plt.ylabel('Predicted IMDb Scores')
plt.title('Random Forest Regression (Description Sentiment)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_holdout_title, alpha=0.5)
plt.xlabel('Actual IMDb Scores')
plt.ylabel('Predicted IMDb Scores')
plt.title('Random Forest Regression (Title Sentiment)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f'Description Sentiment - Mean Squared Error (Holdout): {mse_holdout_description:.2f}')
print(f'Description Sentiment - R-squared (Holdout): {r2_holdout_description:.2f}')
print(f'Title Sentiment - Mean Squared Error (Holdout): {mse_holdout_title:.2f}')
print(f'Title Sentiment - R-squared (Holdout): {r2_holdout_title:.2f}')

print(f'Description Sentiment - Mean Squared Error (k-fold): {mean_mse_cv_description:.2f}')
print(f'Description Sentiment - R-squared (k-fold): {mean_r2_cv_description:.2f}')
print(f'Title Sentiment - Mean Squared Error (k-fold): {mean_mse_cv_title:.2f}')
print(f'Title Sentiment - R-squared (k-fold): {mean_r2_cv_title:.2f}')
