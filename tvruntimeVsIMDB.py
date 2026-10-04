import random
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# linear regression models that which compares runtime and IMDB Score
# helps determine whether longer or shorter movies are more well-received

data = pd.read_csv('Datasets/preprocessed_data.csv')
tv_data = data[data['type'] == 'SHOW']

tv_data = tv_data.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(0.8 * len(tv_data))

tv_train = tv_data[:split_index] 
tv_test = tv_data[split_index:]


X_COL = 'runtime'
Y_COL = 'imdb_score'

X_train = tv_train[X_COL].values.reshape(-1,1)
Y_train = tv_train[Y_COL]

X_test = tv_test[X_COL].values.reshape(-1,1)
Y_test = tv_test[Y_COL]

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print(mse)

plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='red')
# plt.plot(X_test_mr, y_ped_mr, color='red')
plt.title('Runtime vs IMDB Score for TV SHOWS')
plt.xlabel('Runtime')
ax= plt.gca()
ax.set_ylim(0,10)
ax.set_xlim(0,100)

plt.ylabel('IMDB Score')

plt.show()