# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:43:32 2019

@author: kchakraborty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data_train = pd.read_excel("Data_Train.xlsx")
data_test = pd.read_excel("Data_Test.xlsx")

X = data_train.iloc[:, :-1].to_numpy()
y = data_train.iloc[:, 8].to_numpy()

X_t = data_test.iloc[:, :-1].to_numpy()

#converting to string to use string operation for removing ₹ from X and y
X = X.astype(np.str)
X[:,3] = np.char.strip(X[:,3], chars ='₹')
X[:,3] = np.char.replace(X[:,3], ',', '')
X[:,4] = np.char.strip(X[:,4], chars ='₹')
X[:,3] = np.char.replace(X[:,3], 'for', X[:,4])
X[:,5] = np.char.replace(X[:,5], 'NEW', '0')
X[:,5] = np.char.replace(X[:,5], '-', '0')
X[:,6] = np.char.replace(X[:,6], '-', '0')
X[:,7] = np.char.replace(X[:,7], '-', '0')

y = y.astype(np.str)
y = np.char.strip(y, chars ='minutes') 
y = np.char.strip(y, chars =' ')
y = np.array(y, dtype='int')

#removing id

def max_features_in_single_row(data, delimiter):
  max_info = 0 
  for i in data:
    if len(i.split("{}".format(delimiter))) > max_info:
      max_info = len(i.split("{}".format(delimiter)))
  print("\n","-"*35)    
  print("Max_Features in One Observation = ", max_info)
  return max_info

def feature_splitter(feat, name, delimiter, max_info):
  item_lis = list(feat)
  extracted_features = {}

  for i in range(max_info):
    extracted_features['{}_Feature_{}'.format(name, i+1)] = []
  
  print("-"*35)
  print("Features Dictionary : ", extracted_features)

  for i in tqdm(range(len(item_lis))):
    for j in range(max_info):  
      try:
        extracted_features['{}_Feature_{}'.format(name,j+1)].append(item_lis[i].split("{}".format(delimiter))[j].lower().strip())
      except: 
        extracted_features['{}_Feature_{}'.format(name, j+1)].append(np.nan)


  return extracted_features

loc_max = max_features_in_single_row(X[:, 1], ',')
location_splits = feature_splitter(X[:, 1], 'Location', ',', loc_max)

cousine_max = max_features_in_single_row(X[:, 2], ',')
cousine_splits = feature_splitter(X[:, 2], 'Cousine', ',', cousine_max)

X = np.append(X, np.matrix(location_splits["Location_Feature_1"]).transpose(), axis=1)
X = np.append(X, np.matrix(location_splits["Location_Feature_2"]).transpose(), axis=1)
X = np.append(X, np.matrix(location_splits["Location_Feature_3"]).transpose(), axis=1)
X = np.append(X, np.matrix(location_splits["Location_Feature_4"]).transpose(), axis=1)

X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_1"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_2"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_3"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_4"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_5"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_6"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_7"]).transpose(), axis=1)
X = np.append(X, np.matrix(cousine_splits["Cousine_Feature_8"]).transpose(), axis=1)

X = np.delete(X, 1, 1)
X = np.delete(X, 1, 1)

from sklearn.preprocessing import LabelEncoder

labelencoder_0 = LabelEncoder().fit(X[:, 0])
X[:, 0] = np.matrix(labelencoder_0.transform(X[:, 0])).transpose()

labelencoder_loc_1 = LabelEncoder().fit(X[:, 6])
X[:, 6] = np.matrix(labelencoder_loc_1.transform(X[:, 6])).transpose()

labelencoder_loc_2 = LabelEncoder().fit(X[:, 7])
X[:, 7] = np.matrix(labelencoder_loc_2.transform(X[:, 7])).transpose()

labelencoder_loc_3 = LabelEncoder().fit(X[:, 8])
X[:, 8] = np.matrix(labelencoder_loc_3.transform(X[:, 8])).transpose()

labelencoder_loc_4 = LabelEncoder().fit(X[:, 9])
X[:, 9] = np.matrix(labelencoder_loc_4.transform(X[:, 9])).transpose()

labelencoder_cou_1 = LabelEncoder().fit(X[:, 10])
X[:, 10] = np.matrix(labelencoder_cou_1.transform(X[:, 10])).transpose()

labelencoder_cou_2 = LabelEncoder().fit(X[:, 11])
X[:, 11] = np.matrix(labelencoder_cou_2.transform(X[:, 11])).transpose()

labelencoder_cou_3 = LabelEncoder().fit(X[:, 12])
X[:, 12] = np.matrix(labelencoder_cou_3.transform(X[:, 12])).transpose()

labelencoder_cou_4 = LabelEncoder().fit(X[:, 13])
X[:, 13] = np.matrix(labelencoder_cou_4.transform(X[:, 13])).transpose()

labelencoder_cou_5 = LabelEncoder().fit(X[:, 14])
X[:, 14] = np.matrix(labelencoder_cou_5.transform(X[:, 14])).transpose()

labelencoder_cou_6 = LabelEncoder().fit(X[:, 15])
X[:, 15] = np.matrix(labelencoder_cou_6.transform(X[:, 15])).transpose()

labelencoder_cou_7 = LabelEncoder().fit(X[:, 16])
X[:, 16] = np.matrix(labelencoder_cou_7.transform(X[:, 16])).transpose()

labelencoder_cou_8 = LabelEncoder().fit(X[:, 17])
X[:, 17] = np.matrix(labelencoder_cou_8.transform(X[:, 17])).transpose()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))

# Adding the second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
