#%%
# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn import decomposition, preprocessing
from sklearn.model_selection import cross_val_score

#%%
# load data
data = pd.read_csv('../data/preprocessed_data/train_data.csv')
x_data = data.drop(columns = ['cust_no', 'label'])
x_data.columns

# 적합성 검정
# Bartlett's Test, KMO Test(0.6 이상)

# X data
x_X_data = x_data[['X1_m1', 'X2_m1', 'X3_m1', 'X4_m1', 'X5_m1', 'X6_m1', 'X7_m1', 'X8_m1',
        'X1_m2', 'X2_m2', 'X3_m2', 'X4_m2', 'X5_m2', 'X6_m2', 'X7_m2', 'X8_m2',
        'X1_m3', 'X2_m3', 'X3_m3', 'X4_m3', 'X5_m3', 'X6_m3', 'X7_m3', 'X8_m3']]
chi_square_value, p_value = calculate_bartlett_sphericity(x_X_data)
print(chi_square_value, p_value) # O
kmo_all, kmo_model = calculate_kmo(x_X_data)
print(kmo_model) # O

# B data
x_B_data = x_data[['B1_m1', 'B2_m1', 'B3_m1', 'B4_m1', 'B5_m1',
                   'B1_m2', 'B2_m2', 'B3_m2', 'B4_m2', 'B5_m2',
                   'B1_m3', 'B2_m3', 'B3_m3', 'B4_m3', 'B5_m3']]
chi_square_value, p_value = calculate_bartlett_sphericity(x_B_data)
print(chi_square_value, p_value) # O
kmo_all, kmo_model = calculate_kmo(x_B_data)
print(kmo_model) # O

# C data
x_C_data = x_data[['C1_m1', 'C2_m1','C1_m2', 'C2_m2', 'C1_m3', 'C2_m3']]
chi_square_value, p_value = calculate_bartlett_sphericity(x_C_data)
print(chi_square_value, p_value) # O
kmo_all, kmo_model = calculate_kmo(x_C_data)
print(kmo_model) # O

# occupation
x_OCC_data = x_data[['군인', '농축업', '무직', '사무원', '생산직', '서비스직', '은퇴', '전문직', '정치인', '판매원', '프리랜서']]
chi_square_value, p_value = calculate_bartlett_sphericity(x_OCC_data)
print(chi_square_value, p_value) # O
kmo_all, kmo_model = calculate_kmo(x_OCC_data)
print(kmo_model) # X

# E data
x_E_data = x_data[['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E10', 'E14', 'E15', 'E16', 'E17', 'E18']]
chi_square_value, p_value = calculate_bartlett_sphericity(x_E_data)
print(chi_square_value, p_value) # O
kmo_all, kmo_model = calculate_kmo(x_E_data)
print(kmo_model) # X


#%%
# Create factor analysis object and check the Eigenvalues

# X data
fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation='varimax')
fa.fit(x_X_data)
ev, v = fa.get_eigenvalues()
print(ev) # 8

# Create scree plot using matplotlib
plt.scatter(range(1,x_X_data.shape[1]+1),ev)
plt.plot(range(1,x_X_data.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel("‘Factors’")
plt.ylabel("‘Eigenvalue’")
plt.grid()
plt.show() # 7

# B data
fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation='varimax')
fa.fit(x_B_data)
ev, v = fa.get_eigenvalues()
print(ev) # 5

# Create scree plot using matplotlib
plt.scatter(range(1,x_B_data.shape[1]+1),ev)
plt.plot(range(1,x_B_data.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show() # 5

# C data
fa = FactorAnalyzer()
fa.set_params(n_factors=2, rotation='varimax')
fa.fit(x_C_data)
ev, v = fa.get_eigenvalues()
print(ev) # 2

# Create scree plot using matplotlib
plt.scatter(range(1, x_C_data.shape[1]+1), ev)
plt.plot(range(1, x_C_data.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show() # 2

#%%
# create factor analysis object and perform factor analysis
# X data
fa = FactorAnalyzer()
fa.set_params(n_factors=7, rotation='varimax')
fa.fit(x_X_data)
X_loading_df = pd.DataFrame(fa.loadings_, index = x_X_data.columns, columns=['X_Factor1', 'X_Factor2', 'X_Factor3', 'X_Factor4', 'X_Factor5', 'X_Factor6', 'X_Factor7'])
pd.DataFrame(fa.get_factor_variance(), index = ['SS Loadings', 'Proportion Var', 'Cumulative Var'], columns = ['X_Factor1', 'X_Factor2', 'X_Factor3', 'X_Factor4', 'X_Factor5', 'X_Factor6', 'X_Factor7'])
X_score_df = pd.DataFrame(fa.transform(x_X_data), columns=['X_Factor1', 'X_Factor2', 'X_Factor3', 'X_Factor4', 'X_Factor5', 'X_Factor6', 'X_Factor7'])

# B data
fa = FactorAnalyzer()
fa.set_params(n_factors=5, rotation='varimax')
fa.fit(x_B_data)
B_loading_df = pd.DataFrame(fa.loadings_, index = x_B_data.columns, columns=['B_Factor1', 'B_Factor2', 'B_Factor3', 'B_Factor4', 'B_Factor5'])
pd.DataFrame(fa.get_factor_variance(), index = ['SS Loadings', 'Proportion Var', 'Cumulative Var'], columns = ['B_Factor1', 'B_Factor2', 'B_Factor3', 'B_Factor4', 'B_Factor5'])
B_score_df = pd.DataFrame(fa.transform(x_B_data), columns=['B_Factor1', 'B_Factor2', 'B_Factor3', 'B_Factor4', 'B_Factor5'])

# C data
fa = FactorAnalyzer()
fa.set_params(n_factors=2, rotation='varimax')
fa.fit(x_C_data)
C_loading_df = pd.DataFrame(fa.loadings_, index = x_C_data.columns, columns=['C_Factor1', 'C_Factor2'])
pd.DataFrame(fa.get_factor_variance(), index = ['SS Loadings', 'Proportion Var', 'Cumulative Var'], columns = ['C_Factor1', 'C_Factor2'])
C_score_df = pd.DataFrame(fa.transform(x_C_data), columns=['C_Factor1', 'C_Factor2'])

#%%
# merge score data
X_score_df = X_score_df.set_index(x_X_data.index)
B_score_df = B_score_df.set_index(x_B_data.index)
C_score_df = C_score_df.set_index(x_C_data.index)
score_df = pd.concat([X_score_df, B_score_df, C_score_df], axis=1)


new_df = x_data[['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E10', 'E14', 'E15', 'E16', 'E17',
                 'E18', 'I1', 'I2', 'I3', 'I4', 'I6', 'I7', 'I11', 'I15', 'I16', 'I17',
                 'I18', 'I19', 'I20', '군인', '농축업', '무직', '사무원', '생산직', '서비스직',
                 '은퇴', '전문직', '정치인', '판매원', '프리랜서']]


new_df = pd.concat([new_df, score_df, data[['label']]], axis=1)
print(new_df.shape)

#%%
# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

# 1. original data
x_data = data.drop(['cust_no', 'label'], axis=1)
y_data = data['label']
print(x_data.shape, y_data.shape)

x_data.fillna(0, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(x_train, y_train)
end_time = time.time()
print('걸린 시간 :', end_time - start_time)
y_pred = rf.predict(x_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# acc : 0.71, f1-score : 0.57


# 2. factor analysis data
x_data = new_df.drop(['label'], axis=1)
y_data = new_df['label']
print(x_data.shape, y_data.shape)

x_data.fillna(0, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(x_train, y_train)
end_time = time.time()
print('걸린 시간 :', end_time - start_time)
y_pred = rf.predict(x_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# acc : 0.70, f1-score : 0.54