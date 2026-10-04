# import pandas as pd
# import seaborn as sns

# import matplotlib.pyplot as plt

# df = pd.read_csv('Datasets/preprocessed_data.csv')

# description_score = df['sentiment_score_description']
# title_score = df['sentiment_score_titles']
# imdb_score = df['imdb_score']

# plt.figure(figsize=(15, 8))

# plt.scatter(description_score, imdb_score, label='Sentiment Description vs IMDB Score')

# plt.xlabel('Sentiment Score (Description)')
# plt.ylabel('IMDB Score')

# plt.title('Sentiment Description Score vs IMDb Score')
# plt.legend()
# plt.grid(True)
# plt.show()

# # plt.savefig('Description vs IMDB.png')

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('Datasets/preprocessed_data.csv')

# # description_score = df['sentiment_score_description']
# title_score = df['sentiment_score_titles']
# imdb_score = df['imdb_score']

# plt.figure(figsize=(15, 8))

# plt.scatter(title_score, imdb_score, label='Sentiment Title vs IMDB Score')

# plt.xlabel('Sentiment Score (Title)')
# plt.ylabel('IMDB Score')

# plt.title('Sentiment Title Score vs IMDb Score')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.savefig('Description vs IMDB.png')
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('Datasets/preprocessed_data.csv')

description_score = df['sentiment_score_description']
title_score = df['sentiment_score_titles']
imdb_score = df['imdb_score']

# zero is netural
threshold = 0 

positive_description = df[description_score > threshold]['imdb_score']
negative_description = df[description_score <= threshold]['imdb_score']

positive_title = df[title_score > threshold]['imdb_score']
negative_title = df[title_score <= threshold]['imdb_score']

# doing the t-test
t_stat_description, p_value_description = stats.ttest_ind(positive_description, negative_description)
t_stat_title, p_value_title = stats.ttest_ind(positive_title, negative_title)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(positive_description, label='Positive Sentiment', kde=True)
sns.histplot(negative_description, label='Negative Sentiment', kde=True)
plt.title('IMDb Scores by Description Sentiment')
plt.xlabel('IMDb Score')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(positive_title, label='Positive Sentiment', kde=True)
sns.histplot(negative_title, label='Negative Sentiment', kde=True)
plt.title('IMDb Scores by Title Sentiment')
plt.xlabel('IMDb Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()