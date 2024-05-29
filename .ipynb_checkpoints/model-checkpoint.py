import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('train.csv')
df.fillna(df.mean(), inplace=True)

# Assuming '28' is the target variable
X = df.drop('28', axis=1)
y = df['28']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Initialize and train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict on the training set
y_lr_train_pred = lr.predict(x_train)

# Evaluate the model on the training set (you should also evaluate on the test set)
mse_train = mean_squared_error(y_train, y_lr_train_pred)
print("Mean Squared Error on Training Set:", mse_train)
r2_train = lr.score(x_train, y_train)
print("R^2 on Training Set:", r2_train)

# Predict on the test set
# y_lr_test_pred = lr.predict(x_test)

# Optionally, evaluate the model on the test set
# mse_test = mean_squared_error(y_test, y_lr_test_pred)
# print("Mean Squared Error on Test Set:", mse_test)

# Optionally, save the results to a CSV file
label_result_lr = y_lr_train_pred.astype(int)
index_result_lr = df.loc[x_train.index, 'Index'].astype(int)
result_lr = pd.DataFrame({'Index': index_result_lr, 'Label': label_result_lr})
result_lr.to_csv('result_LR.csv', index=False)


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(max_depth=100, random_state=100)
rf.fit(x_train, y_train)
y_rf_train_pred = rf.predict(x_train)
mse_train = mean_squared_error(y_train, y_rf_train_pred)
print("Mean Squared Error on Training Set:", mse_train)
r2_train = rf.score(x_train, y_train)
print("R^2 on Training Set:", r2_train)
label_result_rf = y_rf_train_pred.astype(int)
index_result_rf = df.loc[x_train.index, 'Index'].astype(int)
result_rf = pd.DataFrame({'Index': index_result_rf, 'Label': label_result_rf})
result_rf.to_csv('result_RF.csv', index=False)