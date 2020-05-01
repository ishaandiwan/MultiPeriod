import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load training data
dataset_train = pd.read_csv('ko_train.csv')

# Load test data
dataset_test = pd.read_csv('ko_test.csv')

# Seperate X and Y data
X_train = dataset_train.drop(['Ticker', 'Period Ending', 'Share Price', 'Return'], axis=1)
y_train = dataset_train[['Return']]

X_test = dataset_test.drop(['Ticker', 'Period Ending', 'Share Price', 'Return'], axis=1)
y_test = dataset_test[['Return']]


# fit model on training data
model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Test MSE: %.3f" % mse)
print("Test RMSE: %.3f" % np.sqrt(mse))


dataset_test = np.flip(np.transpose(np.array(dataset_test[['Period Ending']]))[0])

y_pred = np.flip(y_pred)

y_test = np.flip(np.transpose(np.array(y_test))[0])

plt.plot(dataset_test, y_pred, label='Predicted Return')
plt.plot(dataset_test, y_test, label='Actual Return')
plt.xlabel('Date')
plt.ylabel('Predicted Return')
plt.title('Predicted Return v.s. Actual Return')
plt.legend()
plt.show()


