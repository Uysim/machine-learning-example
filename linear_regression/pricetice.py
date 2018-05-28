import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# read CSV file directly from a URL and save the results
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print "=== first 5 rows ==="
print data.head()
print "===================="

# display the last 5 rows
print "=== last 5 rows ==="
print data.tail()
print "==================="

print "Shape: {}".format(data.shape)
feature_cols  = ['TV','radio','newspaper']
result_cols   = 'sales'

# Virtaulize data
# sns.pairplot(data, x_vars=feature_cols, y_vars=result_cols, size=7, aspect=0.7, kind='reg')
# plt.show()
# ===============

features  = data[feature_cols]
results   = data[result_cols]

x_train, x_test, y_train, y_test = train_test_split(features, results)

linreg = LinearRegression()

linreg.fit(x_train, y_train)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


y_pred = linreg.predict(x_test)

# MAE of predictions
mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)

# MSE of predictions
mean_squared_error = metrics.mean_squared_error(y_test, y_pred)

#RMSE of predictions
main_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("mean_absolute_error: {}".format(mean_absolute_error))
print("mean_squared_error: {}".format(mean_squared_error))
print("main_squared_error: {}".format(main_squared_error))
