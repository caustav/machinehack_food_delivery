import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_excel("Data_Train.xlsx")
data_test = pd.read_excel("Data_Test.xlsx")

X = data_train.iloc[:, :-1].to_numpy()
y = data_train.iloc[:, 8].to_numpy()

X_t = data_test.iloc[:, :-1].to_numpy()

#converting to string to use string operation for removing ₹ from X and y
X = np.array(X, dtype='<U5')
X[:,3] = np.char.strip(X[:,3], chars ='₹')
X[:,3] = np.char.replace(X[:,3], ',', '')
X[:,4] = np.char.strip(X[:,4], chars ='₹')
X[:,3] = np.char.replace(X[:,3], 'for', X[:,4])
X[:,5] = np.char.replace(X[:,5], 'NEW', '0')
X[:,5] = np.char.replace(X[:,5], '-', '0')
X[:,6] = np.char.replace(X[:,6], '-', '0')
X[:,7] = np.char.replace(X[:,7], '-', '0')

y = np.array(y, dtype='<U5')
y = np.char.strip(y, chars ='minutes') 
y = np.char.strip(y, chars =' ')
y = np.array(y, dtype='int')

#removing id
X = X[:, 1:]

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)