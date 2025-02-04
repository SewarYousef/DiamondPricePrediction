import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# to see all columns of the data frame in the Run window
pd.set_option('display.width', 320, 'display.max_columns', 12)

# Data Discovery and Visualization
data = pd.read_csv(r"C:\Users\sewar\OneDrive\Desktop\shAI Club\datasets\diamonds.csv", index_col=None, na_values=['NA'],
                   sep=',', low_memory=False)

# Delete all values where any of the dimensions is 0. As it will affect the size attribute
data.drop(data[(data['x'] == 0) | (data['y'] == 0) | (data['z'] == 0)].index, axis=0, inplace=True)

# Define a new size attribute which is more correlated with the final price than the dimensions separated.
data['size'] = data['x'] * data['y'] * data['z']

# Remove the separated dimensions. Because now they are just a duplicate of the size attribute and they are highly
# correlated with it
data.drop(['x', 'y', 'z'], axis=1, inplace=True)

# Remove the index column because it has a significant relation with the price numerically. And we don't want the ML
# algorithm to get confused with that.
data.drop('Unnamed: 0', axis=1, inplace=True)

# Encoding categorical attributes
'''
1- For the cut attribute, we realised that it is not really a categorical attribute. It represents the quality of
   the cut, so we decided to transform its values to [1, 2, 3, 4, ...] instead of a matrix of zeros and ones. So the
   algorithm won't treat them like absolute categories.
'''
encoder = LabelEncoder()
encodedCut = encoder.fit_transform(data['cut'].sort_values())
encodedCutSeries = pd.Series(encodedCut, name='cutNew')
'''
2- The other attributes can be treated like absolute categories so we will just encode them to zeros and ones
'''
color = pd.get_dummies(data['color'], drop_first=True)
cut = pd.get_dummies(data['cut'], drop_first=True)
clarity = pd.get_dummies(data['clarity'], drop_first=True)
data = pd.concat([data.reset_index(drop=True),
                  color.reset_index(drop=True),
                  encodedCutSeries.reset_index(drop=True),
                  clarity.reset_index(drop=True)], axis=1)
data.drop(['color', 'cut', 'clarity'], axis=1, inplace=True)

x = data[['carat', 'depth', 'table', 'size', 'E', 'F', 'G', 'H', 'I', 'J', 'cutNew', 'IF', 'SI1', 'SI2', 'VS1', 'VS2',
          'VVS1', 'VVS2']]
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)


def display_scores(scores, char):
    print(char)
    print('scores:', scores)
    print('Mean:', scores.mean())
    print('Standered deviation:', scores.std())


# Creating an object of LinearRegression model
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

lm = LinearRegression()
# fit model using training data
lm.fit(x_train, y_train)
# correlation
print('Correlation for LinearRegression:', lm.coef_, '\n')
# intercept
print('intercept for LinearRegression:', lm.intercept_, '\n \n')
# Make predictions
predictions = lm.predict(x_val)
# scatterplot of carat vs predicted price
plt.scatter(y_val, predictions)
# plt.show()
'''
K-fold cross-validation: it randomly splits the training set into 10 distinct subsets called folds,
then it trains and evaluates the model 10 times, picking a different fold for evaluation
every time and training on the other 9 folds.
'''
lm_scores = cross_val_score(lm, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
lm_rmse_scores = np.sqrt(-lm_scores)
display_scores(lm_rmse_scores, 'LinearRegression Model =')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')


# Creating an object of Lasso model
lso = Lasso()
lso.fit(x_train, y_train)

predictions = lso.predict(x_val)
lso_mse = mean_squared_error(y_val, predictions)
plt.scatter(y_val, predictions)
plt.xlabel('y validation')
plt.ylabel('predictions')
plt.title('Lasso model plot')
plt.show()

lso_scores = cross_val_score(lso, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
lso_rmse_scores = np.sqrt(-lso_scores)
display_scores(lso_rmse_scores, 'Lasso Model =')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')


# Creating an object of ElasticNet model
els = ElasticNet()
els.fit(x_train, y_train)

predictions = els.predict(x_val)
els_mse = mean_squared_error(y_val, predictions)
plt.scatter(y_val, predictions)
# plt.show()

els_scores = cross_val_score(els, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
els_rmse_scores = np.sqrt(-els_scores)
display_scores(els_rmse_scores, 'ElasticNet Model =')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')


# Creating an object of RidgeRegression model
rdg = Ridge()
rdg.fit(x_train, y_train)

predictions = rdg.predict(x_val)
rdg_mse = mean_squared_error(y_val, predictions)
plt.scatter(y_val, predictions)
# plt.show()

rdg_scores = cross_val_score(rdg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
rdg_rmse_scores = np.sqrt(-rdg_scores)
display_scores(rdg_rmse_scores, 'RidgeRegression Model =')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')


# Creating an object of DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(x_train, y_train)

predictions = tree.predict(x_val)
tree_mse = mean_squared_error(y_val, predictions)
plt.scatter(y_val, predictions, cmap='hot')
# plt.show()

tree_scores = cross_val_score(tree, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores, 'DecisionTreeRegressor Model =')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')
########################################################################
#
# # Creating an object of RandomForestRegression model
# from sklearn.ensemble import RandomForestRegressor
#
# forest = RandomForestRegressor()
# forest.fit(x_train, y_train)
#
# predictions = forest.predict(x_val)
# forest_mse = mean_squared_error(y_val, predictions)
# plt.scatter(y_val, predictions)
# plt.show()
#
# forest_scores = cross_val_score(forest, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores, 'RandonForestRegression Model =')
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')
#

# Creating an object of SupportVectorMachines(SVM) model
# from sklearn.svm import SVR
#
# svr = SVR()
# svr.fit(x_train, y_train)
#
# predictions = svr.predict(x_val)
# svr_mse = mean_squared_error(y_val, predictions)
# plt.scatter(y_val, predictions)
# # plt.show()
#
# svr_scores = cross_val_score(svr, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
# svr_rmse_scores = np.sqrt(-svr_scores)
# display_scores(svr_rmse_scores, 'SupportVectorMachines(SVM) Model = ')
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')
#
#
# # Creating an object of K-nearestNeighbors(KNN) model
# from sklearn.neighbors import KNeighborsRegressor
#
# knn = KNeighborsRegressor()
# knn.fit(x_train, y_train)
#
# predictions = knn.predict(x_val)
# knn_mse = mean_squared_error(y_val, predictions)
# plt.scatter(y_val, predictions)
# # plt.show()
#
# knn_scores = cross_val_score(knn, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
# knn_rmse_scores = np.sqrt(-knn_scores)
# display_scores(knn_rmse_scores, 'K-nearestNeighbors(KNN) Model =')
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)), '\n \n')


# # Regression Evaluation Metrics
# print('MAE:', metrics.mean_absolute_error(y_val, predictions))
# print('MSE:', metrics.mean_squared_error(y_val, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)))
