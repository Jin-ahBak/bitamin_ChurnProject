import pandas as pd
import numpy as np

#%% Load data
X_test = pd.read_csv('./data/X_test.csv')
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')

#%% show data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

print(X_train.head())
print(y_train.head())
print(X_test.head())

