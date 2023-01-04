import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
aum_m7 = pd.read_csv('../data/rawdata/aum_train/aum_m7.csv')
aum_m8 = pd.read_csv('../data/rawdata/aum_train/aum_m8.csv')
aum_m9 = pd.read_csv('../data/rawdata/aum_train/aum_m9.csv')
aum_m10 = pd.read_csv('../data/rawdata/aum_train/aum_m10.csv')
aum_m11 = pd.read_csv('../data/rawdata/aum_train/aum_m11.csv')
aum_m12 = pd.read_csv('../data/rawdata/aum_train/aum_m12.csv')

behavior_m7 = pd.read_csv('../data/rawdata/behavior_train/behavior_m7.csv')
behavior_m8 = pd.read_csv('../data/rawdata/behavior_train/behavior_m8.csv')
behavior_m9 = pd.read_csv('../data/rawdata/behavior_train/behavior_m9.csv')
behavior_m10 = pd.read_csv('../data/rawdata/behavior_train/behavior_m10.csv')
behavior_m11 = pd.read_csv('../data/rawdata/behavior_train/behavior_m11.csv')
behavior_m12 = pd.read_csv('../data/rawdata/behavior_train/behavior_m12.csv')

big_event_Q3 = pd.read_csv('../data/rawdata/big_event_train/big_event_Q3.csv')
big_event_Q4 = pd.read_csv('../data/rawdata/big_event_train/big_event_Q4.csv')

cunkuan_m7 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m7.csv')
cunkuan_m8 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m8.csv')
cunkuan_m9 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m9.csv')
cunkuan_m10 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m10.csv')
cunkuan_m11 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m11.csv')
cunkuan_m12 = pd.read_csv('../data/rawdata/cunkuan_train/cunkuan_m12.csv')

cust_avli_Q3 = pd.read_csv('../data/rawdata/cust_avil_train/cust_avli_Q3.csv')
cust_avli_Q4 = pd.read_csv('../data/rawdata/cust_avil_train/cust_avli_Q4.csv')

cust_info_q3 = pd.read_csv('../data/rawdata/cust_info_train/cust_info_q3.csv')
cust_info_q4 = pd.read_csv('../data/rawdata/cust_info_train/cust_info_q4.csv')

y_Q3_3 = pd.read_csv('../data/rawdata/train_label/y_Q3_3.csv')
y_Q4_3 = pd.read_csv('../data/rawdata/train_label/y_Q4_3.csv')
#%%

# print data shape
print("aum_m7 shape: ", aum_m7.shape)
print("aum_m8 shape: ", aum_m8.shape)
print("aum_m9 shape: ", aum_m9.shape)
print("aum_m10 shape: ", aum_m10.shape)
print("aum_m11 shape: ", aum_m11.shape)
print("aum_m12 shape: ", aum_m12.shape)

print("behavior_m7 shape: ", behavior_m7.shape)
print("behavior_m8 shape: ", behavior_m8.shape)
print("behavior_m9 shape: ", behavior_m9.shape)
print("behavior_m10 shape: ", behavior_m10.shape)
print("behavior_m11 shape: ", behavior_m11.shape)
print("behavior_m12 shape: ", behavior_m12.shape)

print("big_event_Q3 shape: ", big_event_Q3.shape)
print("big_event_Q4 shape: ", big_event_Q4.shape)

print("cunkuan_m7 shape: ", cunkuan_m7.shape)
print("cunkuan_m8 shape: ", cunkuan_m8.shape)
print("cunkuan_m9 shape: ", cunkuan_m9.shape)
print("cunkuan_m10 shape: ", cunkuan_m10.shape)
print("cunkuan_m11 shape: ", cunkuan_m11.shape)
print("cunkuan_m12 shape: ", cunkuan_m12.shape)

print("cust_avil_Q3 shape: ", cust_avli_Q3.shape)
print("cust_avil_Q4 shape: ", cust_avli_Q4.shape)

print("cust_info_q3 shape: ", cust_info_q3.shape)
print("cust_info_q4 shape: ", cust_info_q4.shape)

print("y_Q3_3 shape: ", y_Q3_3.shape)
print("y_Q4_3 shape: ", y_Q4_3.shape)

#%%
# y train data
y_Q3_3.rename(columns={'cust_no':'cust_no', 'label':'label_Q3'}, inplace=True)
y_Q4_3.rename(columns={'cust_no':'cust_no', 'label':'label_Q4'}, inplace=True)
y_train = pd.merge(y_Q3_3, y_Q4_3, on='cust_no', how='outer')
y_train_idx = y_train.cust_no

# aum_m[7,8,9,10,11,12].cust_no 이 y_train_idx에 존재하는 데이터만 추출
aum_m7 = aum_m7[aum_m7.cust_no.isin(y_train_idx)]
aum_m8 = aum_m8[aum_m8.cust_no.isin(y_train_idx)]
aum_m9 = aum_m9[aum_m9.cust_no.isin(y_train_idx)]
aum_m10 = aum_m10[aum_m10.cust_no.isin(y_train_idx)]
aum_m11 = aum_m11[aum_m11.cust_no.isin(y_train_idx)]
aum_m12 = aum_m12[aum_m12.cust_no.isin(y_train_idx)]

# behavior_m[7,8,9,10,11,12].cust_no 이 y_train_idx에 존재하는 데이터만 추출
behavior_m7 = behavior_m7[behavior_m7.cust_no.isin(y_train_idx)]
behavior_m8 = behavior_m8[behavior_m8.cust_no.isin(y_train_idx)]
behavior_m9 = behavior_m9[behavior_m9.cust_no.isin(y_train_idx)]
behavior_m10 = behavior_m10[behavior_m10.cust_no.isin(y_train_idx)]
behavior_m11 = behavior_m11[behavior_m11.cust_no.isin(y_train_idx)]
behavior_m12 = behavior_m12[behavior_m12.cust_no.isin(y_train_idx)]

# big_event_Q[3,4].cust_no 이 y_train_idx에 존재하는 데이터만 추출
big_event_Q3 = big_event_Q3[big_event_Q3.cust_no.isin(y_train_idx)]
big_event_Q4 = big_event_Q4[big_event_Q4.cust_no.isin(y_train_idx)]

# cunkuan_m[7,8,9,10,11,12].cust_no 이 y_train_idx에 존재하는 데이터만 추출
cunkuan_m7 = cunkuan_m7[cunkuan_m7.cust_no.isin(y_train_idx)]
cunkuan_m8 = cunkuan_m8[cunkuan_m8.cust_no.isin(y_train_idx)]
cunkuan_m9 = cunkuan_m9[cunkuan_m9.cust_no.isin(y_train_idx)]
cunkuan_m10 = cunkuan_m10[cunkuan_m10.cust_no.isin(y_train_idx)]
cunkuan_m11 = cunkuan_m11[cunkuan_m11.cust_no.isin(y_train_idx)]
cunkuan_m12 = cunkuan_m12[cunkuan_m12.cust_no.isin(y_train_idx)]

# cust_avli_Q[3,4].cust_no 이 y_train_idx에 존재하는 데이터만 추출
cust_avli_Q3 = cust_avli_Q3[cust_avli_Q3.cust_no.isin(y_train_idx)]
cust_avli_Q4 = cust_avli_Q4[cust_avli_Q4.cust_no.isin(y_train_idx)]

# cust_info_q[3,4].cust_no 이 y_train_idx에 존재하는 데이터만 추출
cust_info_q3 = cust_info_q3[cust_info_q3.cust_no.isin(y_train_idx)]
cust_info_q4 = cust_info_q4[cust_info_q4.cust_no.isin(y_train_idx)]


#%%

# data merge

# 1. aum_m[7,8,9,10,11,12] merge , column name : real colname + m[7,8,9,10,11,12]

# rename for merge
aum_m7 = aum_m7.rename(columns={'X1':'X1_m7', 'X2':'X2_m7', 'X3':'X3_m7', 'X4':'X4_m7', 'X5':'X5_m7', 'X6':'X6_m7', 'X7':'X7_m7', 'X8':'X8_m7'})
aum_m8 = aum_m8.rename(columns={'X1':'X1_m8', 'X2':'X2_m8', 'X3':'X3_m8', 'X4':'X4_m8', 'X5':'X5_m8', 'X6':'X6_m8', 'X7':'X7_m8', 'X8':'X8_m8'})
aum_m9 = aum_m9.rename(columns={'X1':'X1_m9', 'X2':'X2_m9', 'X3':'X3_m9', 'X4':'X4_m9', 'X5':'X5_m9', 'X6':'X6_m9', 'X7':'X7_m9', 'X8':'X8_m9'})
aum_m10 = aum_m10.rename(columns={'X1':'X1_m10', 'X2':'X2_m10', 'X3':'X3_m10', 'X4':'X4_m10', 'X5':'X5_m10', 'X6':'X6_m10', 'X7':'X7_m10', 'X8':'X8_m10'})
aum_m11 = aum_m11.rename(columns={'X1':'X1_m11', 'X2':'X2_m11', 'X3':'X3_m11', 'X4':'X4_m11', 'X5':'X5_m11', 'X6':'X6_m11', 'X7':'X7_m11', 'X8':'X8_m11'})
aum_m12 = aum_m12.rename(columns={'X1':'X1_m12', 'X2':'X2_m12', 'X3':'X3_m12', 'X4':'X4_m12', 'X5':'X5_m12', 'X6':'X6_m12', 'X7':'X7_m12', 'X8':'X8_m12'})

# merge
aum_m = pd.merge(aum_m7, aum_m8, on='cust_no', how='outer')
aum_m = pd.merge(aum_m, aum_m9, on='cust_no', how='outer')
aum_m = pd.merge(aum_m, aum_m10, on='cust_no', how='outer')
aum_m = pd.merge(aum_m, aum_m11, on='cust_no', how='outer')
aum_m = pd.merge(aum_m, aum_m12, on='cust_no', how='outer')

del aum_m7, aum_m8, aum_m9, aum_m10, aum_m11, aum_m12

# 2. behavior_m[7,8,9,10,11,12] merge , column name : real colname + m[7,8,9,10,11,12]
# rename for merge
behavior_m7 = behavior_m7.rename(columns={'B1':'B1_m7', 'B2':'B2_m7', 'B3':'B3_m7', 'B4':'B4_m7', 'B5':'B5_m7'})
behavior_m8 = behavior_m8.rename(columns={'B1':'B1_m8', 'B2':'B2_m8', 'B3':'B3_m8', 'B4':'B4_m8', 'B5':'B5_m8'})
behavior_m9 = behavior_m9.rename(columns={'B1':'B1_m9', 'B2':'B2_m9', 'B3':'B3_m9', 'B4':'B4_m9', 'B5':'B5_m9', 'B6':'B6_m9', 'B7':'B7_m9'})
behavior_m10 = behavior_m10.rename(columns={'B1':'B1_m10', 'B2':'B2_m10', 'B3':'B3_m10', 'B4':'B4_m10', 'B5':'B5_m10'})
behavior_m11 = behavior_m11.rename(columns={'B1':'B1_m11', 'B2':'B2_m11', 'B3':'B3_m11', 'B4':'B4_m11', 'B5':'B5_m11'})
behavior_m12 = behavior_m12.rename(columns={'B1':'B1_m12', 'B2':'B2_m12', 'B3':'B3_m12', 'B4':'B4_m12', 'B5':'B5_m12', 'B6':'B6_m12', 'B7':'B7_m12'})

# merge
behavior_m = pd.merge(behavior_m7, behavior_m8, on='cust_no', how='outer')
behavior_m = pd.merge(behavior_m, behavior_m9, on='cust_no', how='outer')
behavior_m = pd.merge(behavior_m, behavior_m10, on='cust_no', how='outer')
behavior_m = pd.merge(behavior_m, behavior_m11, on='cust_no', how='outer')
behavior_m = pd.merge(behavior_m, behavior_m12, on='cust_no', how='outer')

del behavior_m7, behavior_m8, behavior_m9, behavior_m10, behavior_m11, behavior_m12

# 3. cunkuan_m[7,8,9,10,11,12] merge , column name : real colname + m[7,8,9,10,11,12]

# rename for merge
cunkuan_m7 = cunkuan_m7.rename(columns = {'C1':'C1_m7', 'C2':'C2_m7'})
cunkuan_m8 = cunkuan_m8.rename(columns = {'C1':'C1_m8', 'C2':'C2_m8'})
cunkuan_m9 = cunkuan_m9.rename(columns = {'C1':'C1_m9', 'C2':'C2_m9'})
cunkuan_m10 = cunkuan_m10.rename(columns = {'C1':'C1_m10', 'C2':'C2_m10'})
cunkuan_m11 = cunkuan_m11.rename(columns = {'C1':'C1_m11', 'C2':'C2_m11'})
cunkuan_m12 = cunkuan_m12.rename(columns = {'C1':'C1_m12', 'C2':'C2_m12'})

# merge
cunkuan_m = pd.merge(cunkuan_m7, cunkuan_m8, on='cust_no', how='outer')
cunkuan_m = pd.merge(cunkuan_m, cunkuan_m9, on='cust_no', how='outer')
cunkuan_m = pd.merge(cunkuan_m, cunkuan_m10, on='cust_no', how='outer')
cunkuan_m = pd.merge(cunkuan_m, cunkuan_m11, on='cust_no', how='outer')
cunkuan_m = pd.merge(cunkuan_m, cunkuan_m12, on='cust_no', how='outer')

del cunkuan_m7, cunkuan_m8, cunkuan_m9, cunkuan_m10, cunkuan_m11, cunkuan_m12

# 4.big_event_Q[3,4] merge , column name : real colname + Q[3,4]

# rename for merge
big_event_Q3 = big_event_Q3.rename(columns = {'E1':'E1_Q3', 'E2':'E2_Q3', 'E3':'E3_Q3', 'E4':'E4_Q3',
                                              'E5':'E5_Q3', 'E6':'E6_Q3', 'E7':'E7_Q3', 'E8':'E8_Q3',
                                              'E9':'E9_Q3', 'E10':'E10_Q3', 'E11':'E11_Q3', 'E12':'E12_Q3',
                                              'E13':'E13_Q3', 'E14':'E14_Q3', 'E15':'E15_Q3', 'E16':'E16_Q3',
                                              'E17':'E17_Q3', 'E18':'E18_Q3'})
big_event_Q4 = big_event_Q4.rename(columns = {'E1':'E1_Q4', 'E2':'E2_Q4', 'E3':'E3_Q4', 'E4':'E4_Q4',
                                                'E5':'E5_Q4', 'E6':'E6_Q4', 'E7':'E7_Q4', 'E8':'E8_Q4',
                                                'E9':'E9_Q4', 'E10':'E10_Q4', 'E11':'E11_Q4', 'E12':'E12_Q4',
                                                'E13':'E13_Q4', 'E14':'E14_Q4', 'E15':'E15_Q4', 'E16':'E16_Q4',
                                                'E17':'E17_Q4', 'E18':'E18_Q4'})

# merge
big_event_Q = pd.merge(big_event_Q3, big_event_Q4, on='cust_no', how='outer')

del big_event_Q3, big_event_Q4

# 5. cust_info_q[3,4] merge , column name : real colname + Q[3,4]

# rename for merge
cust_info_q3 = cust_info_q3.rename(columns = {'I1':'I1_Q3', 'I2':'I2_Q3', 'I3':'I3_Q3', 'I4':'I4_Q3',
                                    'I5':'I5_Q3', 'I6':'I6_Q3', 'I7':'I7_Q3', 'I8':'I8_Q3',
                                    'I9':'I9_Q3', 'I10':'I10_Q3', 'I11':'I11_Q3', 'I12':'I12_Q3',
                                    'I13':'I13_Q3', 'I14':'I14_Q3', 'I15':'I15_Q3', 'I16':'I16_Q3',
                                    'I17':'I17_Q3', 'I18':'I18_Q3', 'I19':'I19_Q3', 'I20':'I20_Q3'})
cust_info_q4 = cust_info_q4.rename(columns = {'I1':'I1_Q4', 'I2':'I2_Q4', 'I3':'I3_Q4', 'I4':'I4_Q4',
                                    'I5':'I5_Q4', 'I6':'I6_Q4', 'I7':'I7_Q4', 'I8':'I8_Q4',
                                    'I9':'I9_Q4', 'I10':'I10_Q4', 'I11':'I11_Q4', 'I12':'I12_Q4',
                                    'I13':'I13_Q4', 'I14':'I14_Q4', 'I15':'I15_Q4', 'I16':'I16_Q4',
                                    'I17':'I17_Q4', 'I18':'I18_Q4', 'I19':'I19_Q4', 'I20':'I20_Q4'})

# merge
cust_info_q = pd.merge(cust_info_q3, cust_info_q4, on='cust_no', how='outer')

del cust_info_q3, cust_info_q4

#%%
print(aum_m.shape)
print(behavior_m.shape)
print(cunkuan_m.shape)   # 82896
print(big_event_Q.shape)
print(cust_info_q.shape)

#%%

def check_missing(dat):
    '''Print missing values in each column of the dat
    @Param df dat: input data frame
    '''
    missing_val = dat.isnull().sum()
    for index in missing_val.index:
        if missing_val[index] > 0:
            print('{} has {} missing values. ({:.4%})'.format(index, missing_val[index], missing_val[index]/dat.shape[0]))

# aum_m

check_missing(aum_m)
aum_m.isnull().sum()
aum_m[~aum_m.X8_m10.isna()][['X8_m8', 'X8_m9', 'X8_m10', 'X8_m11', 'X8_m12']]
aum_m.fillna(0, inplace=True)

# behavior_m

check_missing(behavior_m)
behavior_m[behavior_m.B5_m10.isna()][['B5_m8', 'B5_m9', 'B5_m10', 'B5_m11', 'B5_m12']]
behavior_m.fillna(0, inplace=True)

# cunkuan_m

check_missing(cunkuan_m)
cunkuan_m[cunkuan_m.C1_m8.isna()][['C1_m8', 'C1_m9', 'C1_m10', 'C1_m11', 'C1_m12']]
cunkuan_m.fillna(0, inplace=True)

# big_event_Q

check_missing(big_event_Q)
big_event_Q.drop(columns=['E1_Q3', 'E2_Q3', 'E3_Q3', 'E4_Q3',
                          'E5_Q3', 'E6_Q3', 'E7_Q3', 'E8_Q3',
                          'E9_Q3', 'E10_Q3', 'E11_Q3', 'E12_Q3',
                          'E13_Q3', 'E14_Q3', 'E15_Q3', 'E16_Q3',
                          'E17_Q3', 'E18_Q3'], inplace=True)

big_event_Q.drop(columns = ['E4_Q4', 'E5_Q4', 'E7_Q4', 'E8_Q4',
                            'E9_Q4', 'E11_Q4', 'E12_Q4', 'E13_Q4',
                            'E14_Q4', 'E16_Q4', 'E18_Q4'], inplace = True)

big_event_Q.shape

# cust_info_q
check_missing(cust_info_q)

cust_info_q.drop(columns = ['I9_Q3', 'I10_Q3', 'I13_Q3', 'I14_Q3',
                            'I9_Q4', 'I10_Q4', 'I13_Q4', 'I14_Q4'], inplace = True)

#cust_info_q[['I1_Q3', 'I1_Q4']][(cust_info_q.I1_Q3.isna()) & (cust_info_q.I1_Q4.isna())].isna().sum()
cust_info_q.drop(columns = ['I1_Q3', 'I2_Q3'], inplace = True)
cust_info_q.rename(columns = {'I1_Q4':'I1', 'I2_Q4':'I2'}, inplace = True)
cust_info_q.drop(columns = ['I8_Q3', 'I8_Q4'], inplace = True)
cust_info_q.drop(columns = ['I12_Q3', 'I12_Q4'], inplace = True)

cust_info_q.isna().sum()

# I3_Q[3,4,5,6,7,11,15,16,17,18,19,20]의 결측을 I3_Q4로 채움
cust_info_q.I3_Q3.fillna(cust_info_q.I3_Q4, inplace=True)
cust_info_q.I4_Q3.fillna(cust_info_q.I4_Q4, inplace=True)
cust_info_q.I5_Q3.fillna(cust_info_q.I5_Q4, inplace=True)
cust_info_q.I6_Q3.fillna(cust_info_q.I6_Q4, inplace=True)
cust_info_q.I7_Q3.fillna(cust_info_q.I7_Q4, inplace=True)
cust_info_q.I11_Q3.fillna(cust_info_q.I11_Q4, inplace=True)
cust_info_q.I15_Q3.fillna(cust_info_q.I15_Q4, inplace=True)
cust_info_q.I16_Q3.fillna(cust_info_q.I16_Q4, inplace=True)
cust_info_q.I17_Q3.fillna(cust_info_q.I17_Q4, inplace=True)
cust_info_q.I18_Q3.fillna(cust_info_q.I18_Q4, inplace=True)
cust_info_q.I19_Q3.fillna(cust_info_q.I19_Q4, inplace=True)
cust_info_q.I20_Q3.fillna(cust_info_q.I20_Q4, inplace=True)


# I5_Q3, I5_Q4의 결측 '무직'으로 대체
cust_info_q.I5_Q3.fillna('무직', inplace=True)
cust_info_q.I5_Q4.fillna('무직', inplace=True)

#%%

print(aum_m.shape)
print(behavior_m.shape)
print(cunkuan_m.shape)
print(big_event_Q.shape)
print(cust_info_q.shape)

#%%

# aum_m, behavior_m, cunkuan_m, big_event_Q, cust_info_q, y_train을 합침
train = pd.merge(aum_m, behavior_m, on='cust_no', how='left')
train = pd.merge(train, cunkuan_m, on='cust_no', how='left')
train = pd.merge(train, big_event_Q, on='cust_no', how='left')
train = pd.merge(train, cust_info_q, on='cust_no', how='left')
train = pd.merge(train, y_train, on='cust_no', how='left')

# C1이 결측인 행 제거
train.dropna(subset=['C1_m11'], inplace=True)

# 결측 확인
check_missing(train)

# 'E1_Q4','E2_Q4' datetime으로 변환
train[['E1_Q4', 'E2_Q4', 'E3_Q4', 'E6_Q4', 'E10_Q4']] = train[['E1_Q4', 'E2_Q4', 'E3_Q4', 'E6_Q4', 'E10_Q4']].apply(pd.to_datetime)

#
# pd.datetime('2019-12-31')

def date_diff(x):
    return (pd.datetime(2019,12,31) - x).days # 4분기 기준으로 날짜 데이터 전처리

train.E1_Q4 = train.E1_Q4.apply(date_diff)
train.E2_Q4 = train.E2_Q4.apply(date_diff)
train.E3_Q4 = train.E3_Q4.apply(date_diff)
train.E6_Q4 = train.E6_Q4.apply(date_diff)
train.E10_Q4 = train.E10_Q4.apply(date_diff)

# E1_Q4, E2_Q4, E3_Q4, E6_Q4, E10_Q4 결측 max값으로 대체
train.E1_Q4.fillna(train.E1_Q4.max(), inplace=True)
train.E2_Q4.fillna(train.E2_Q4.max(), inplace=True)
train.E3_Q4.fillna(train.E3_Q4.max(), inplace=True)
train.E6_Q4.fillna(train.E6_Q4.max(), inplace=True)
train.E10_Q4.fillna(train.E10_Q4.max(), inplace=True)

# int형 변환
train[['E1_Q4', 'E2_Q4', 'E3_Q4', 'E6_Q4', 'E10_Q4']] = train[['E1_Q4', 'E2_Q4', 'E3_Q4', 'E6_Q4', 'E10_Q4']].astype(int)

train.isna().sum()