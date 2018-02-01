import numpy
import lstm
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Assign spreadsheet filename to `file`
file = 'houseprice.xlsx'

# Load spreadsheet
xl = pd.ExcelFile(file)

# Print the sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
houseprice = xl.parse('Sheet1')


#scatter plot grlivarea/saleprice
var = 'YearBuilt'
data = pd.concat([houseprice['HousePrice'], houseprice[var]], axis=1)
data.plot.scatter(x=var, y='HousePrice', ylim=(0,800000));
import seaborn as sns
sns.lmplot(x='SQF', y='HousePrice', data=houseprice)

#box plot overallqual/saleprice
var = 'Bathroom'
data = pd.concat([houseprice['HousePrice'], houseprice[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 10))
fig = sns.boxplot(x=var, y="HousePrice", data=data)
fig.axis(ymin=0, ymax=1800000);

var = 'HouseType'
data = pd.concat([houseprice['HousePrice'], houseprice[var]], axis=1)
f, ax = plt.subplots(figsize=(24, 18))
fig = sns.boxplot(x=var, y="HousePrice", data=data)
fig.axis(ymin=0, ymax=1000000);



#correlation matrix
corrmat = houseprice.iloc[:,[3,4,5,7,8,9,11,12]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);



#Modelling - Artificial Neural Network
#Creating features and target variable
X = houseprice.iloc[:, [ 1,3,4,5,7,8,9,10,11,12]].values
y = houseprice.iloc[:, 6].values

featureNames = ['City', 'Zipcode', 'Longitude', 'Latitude', 
                    'Bedroom', 'Bathroom', 'SQF', 'HouseType', 'RentEstimate', 'YearBuilt']

#Creating dummy variables for categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_gender = LabelEncoder()
X[:, 0] = labelencoder_gender.fit_transform(X[:, 0])

labelencoder_gender = LabelEncoder()
X[:, 7] = labelencoder_gender.fit_transform(X[:, 7])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for City
X = X[:, 1:]


onehotencoder = OneHotEncoder(categorical_features=[41])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for House Type
X = X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#Scaling all features to a similiar scale
#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X, y)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), featureNames), 
             reverse=True))

importances = rf.feature_importances_
indices = numpy.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), featureNames) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(57, input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)



#Support Vector Machines
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())
    
print(rmse(y_rbf, y_test))
print(rmse(y_lin, y_test))
print(rmse(y_poly, y_test))

#Hyperparameter tuning for SVR
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [50, 100, 200, 500, 1000, 1500, 1800, 2000], 'kernel' : ['linear', 'rbf']}]
grid_search = GridSearchCV(estimator = svr_lin, param_grid = parameters, 
                           scoring = 'neg_mean_squared_error',
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#RMSE
numpy.sqrt(abs(best_accuracy))
lw = 2
plt.figure(figsize=(12, 7))
plt.scatter(X_test, y_test, color='darkorange', label='data')
plt.plot(y_test, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(y_test, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(y_test, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vectr polynomial here')
poly = PolynomialFeatures(2)
# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)
# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train)
# get the predictions
y_preds = lin_regressor.predict(poly.fit_transform(X_test))
plt.legend()
plt.show()



#Poly Regression
# create a Linear Regressor   
lin_regressor = LinearRegression()
# pass the order of you
print(rmse(y_preds, poly.fit_transform(y_test)))







