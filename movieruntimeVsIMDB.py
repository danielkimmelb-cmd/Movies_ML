import random
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# linear regression models that which compares runtime and popularity
# helps determine whether longer or shorter movies are more well-received

data = pd.read_csv('everything.csv')
movie_data = data[data['type'] == 'MOVIE']
movie_data = movie_data.sample(frac=1, random_state=42).reset_index(drop=True)


split_index = int(0.8 * len(movie_data))

movie_train = movie_data[:split_index] 
movie_test = movie_data[split_index:]


X_COL = 'runtime'
Y_COL = 'imdb_score'

X_train = movie_train[X_COL].values.reshape(-1,1)
Y_train = movie_train[Y_COL].values

X_test = movie_test[X_COL].values.reshape(-1,1)
Y_test = movie_test[Y_COL]

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print(mse)

plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='red')
# plt.plot(X_test_mr, y_ped_mr, color='red')
plt.title('Runtime vs IMDB Score for Movies')
plt.xlabel('Runtime')
ax= plt.gca()
ax.set_ylim(0,10)
ax.set_xlim(0,240)

plt.ylabel('IMDB Score')

plt.show()