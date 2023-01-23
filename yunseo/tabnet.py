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
    if types[col] == 'object' or nunique[col] < 10:
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
                        cat_emb_dim = 20,
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
    patience=20,
    batch_size=32,
    virtual_batch_size=64,
    num_workers=0,
    weights=1,
    drop_last=False,
)
end_time = time.time()
print("time :", end_time - start)

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