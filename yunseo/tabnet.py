#%%
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

from sklearn.preprocessing import LabelEncoder

#%%
# load data
train_df = pd.read_csv('../data/preprocessed_data/train_data.csv')
valid_df = pd.read_csv('../data/preprocessed_data/valid_data.csv')

x_train = train_df.drop(columns = ['cust_no', 'label'])
y_train = train_df['label']
x_valid = valid_df.drop(columns = ['cust_no', 'label'])
y_valid = valid_df['label']


x_train.fillna(0, inplace=True)
x_valid.fillna(0, inplace=True)

#%%
# data preprocessing
nunique = x_train.nunique()
types = x_valid.dtypes

categorical_columns = []
categorical_dims = {}
for col in x_train.columns:
    if types[col] == 'object' or nunique[col] < 3:
        print(col, x_train[col].nunique())
        l_enc = LabelEncoder()
        x_train[col] = l_enc.fit_transform(x_train[col].values)
        x_valid[col] = l_enc.transform(x_valid[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

print(x_train.shape, x_valid.shape)

# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음
features = [col for col in x_train.columns]
cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

x_train = x_train.values
x_valid = x_valid.values
y_train = y_train.values
y_valid = y_valid.values

print(x_train.shape, x_valid.shape)
print(y_train.shape, y_valid.shape)

#%%
# define model
# clf = TabNetClassifier(cat_idxs = cat_idxs,
#                         cat_dims = cat_dims,
#                         cat_emb_dim = 10,
#                         optimizer_fn = torch.optim.Adam,
#                         optimizer_params = dict(lr=2e-2),
#                         mask_type = 'sparsemax', # 'sparsemax', 'entmax'
#                         scheduler_params = dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
#                         scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau,
#                         verbose = 10,
#                        )

clf = TabNetClassifier(cat_idxs = cat_idxs,
                        cat_dims = cat_dims,
                        # optimizer_fn=torch.optim.Adam,
                        scheduler_params={"step_size":10, "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='sparsemax',
                        cat_emb_dim = 1,
                        # optimizer_params = dict(lr=2e-4),
                        # gamma = 0.7
                        # mask_type = 'entmax', # 'sparsemax', 'entmax'
                        # verbose = 5,
                       )

# fit model
start = time.time()
clf.fit(
    X_train=x_train,
    y_train=y_train,
    eval_set=[(x_train, y_train), (x_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=5,
    batch_size=1024,
    virtual_batch_size=64,
    num_workers=0,
    weights=1,
    drop_last=False,
)
end_time = time.time()
print("time :", end_time - start)



y_pred = clf.predict(x_train)
print('Accuracy: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))

y_pred = clf.predict(x_valid)
print('Accuracy: {:.2f}'.format(accuracy_score(y_valid, y_pred)))
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))



#%%

# # TabNetPretrainer
# unsupervised_model = TabNetPretrainer(
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     mask_type='entmax' # "sparsemax"
# )
#
# unsupervised_model.fit(
#     X_train=X_train,
#     eval_set=[X_valid],
#     pretraining_ratio=0.8,
# )
#
# clf = TabNetClassifier(
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     scheduler_params={"step_size":10, # how to use learning rate scheduler
#                       "gamma":0.9},
#     scheduler_fn=torch.optim.lr_scheduler.StepLR,
#     mask_type='sparsemax' # This will be overwritten if using pretrain model
# )
#
# clf.fit(
#     X_train=X_train, y_train=y_train,
#     eval_set=[(X_train, y_train), (X_valid, y_valid)],
#     eval_name=['train', 'valid'],
#     eval_metric=['auc'],
#     from_unsupervised=unsupervised_model
# )



#%%
# result
# 1. cat_emb_dim = 20, default
# time : 0.0
# 발산
# train_Accuracy: 0.5 , valid_Accuracy: 0.5

# 2. cat_emb_dim = 20, lr = 2e-5
# 1의 결과가 발산하는 것을 보았을 때, lr이 너무 커서 그런가 싶어 lr을 2e-5로 낮춰봄
# 너무 느림
# time : 0.0
# train_Accuracy:  , valid_Accuracy:

# 3. cat_emb_dim = 10, lr - 2e-4, batch_size = 256, virtual_batch_size = 64, gamma = 0.7
# 구림

# 4. default, batch_size = 1024, virtual_batch_size = 64
# time : 0.0
# train_Acc/f1:  0.63, 0.3, valid_Acc/f1:  0.64, 0.3

# 5. default, batch_size = 1024, virtual_batch_size = 64
#    scheduler_params={"step_size":10, "gamma":0.9},# how to use learning rate scheduler
#    scheduler_fn=torch.optim.lr_scheduler.StepLR,
#    mask_type='sparsemax' # This will be overwritten if using pretrain model
#    발산
# loss = 0.9
# train_Acc/f1: , valid_Acc/f1:

# 6. default, batch_size = 1024, virtual_batch_size = 64, cat_emb_dim = 10
#    scheduler_params={"step_size":10, "gamma":0.9},# how to use learning rate scheduler
#    scheduler_fn=torch.optim.lr_scheduler.StepLR,
#    mask_type='sparsemax' # This will be overwritten if using pretrain model
# train_Acc/f1:  0.5 , valid_Acc/f1: 0.5

# 7. category value 기준을 3으로 낮춤. (진짜 cat 변수만 들어가도록)
#    batch_size = 1024, virtual_batch_size = 64, cat_emb_dim = 10
#    scheduler_params={"step_size":10, "gamma":0.9},# how to use learning rate scheduler
#    scheduler_fn=torch.optim.lr_scheduler.StepLR,
#    mask_type='sparsemax' # This will be overwritten if using pretrain model
#    일단 성능이 나아짐. -> 연속형 변수를 cat 변수로 넣었어서 성능이 구렸던 것 같음. (당연함...)
#    train_Acc/f1: 0.63/0.33 , valid_Acc/f1: 0.63/0.33

# 8. 7 + cat_emb_dim = 20
#    train_Acc/f1: , valid_Acc/f1:
#    구려서 멈춤

# 9. 7 + cat_emb_dim = 1
#    train_Acc/f1: 0.57 / 0.48, valid_Acc/f1: 0.57 / 0.48

# -- 일단 tabnet은 여기까지만 하고 다른 모델로 넘어감 --