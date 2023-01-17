import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

# load data
train_df = pd.read_csv('../data/preprocessed_data/train_data.csv')
valid_df = pd.read_csv('../data/preprocessed_data/valid_data.csv')
x_train = train_df.drop(columns = ['cust_no', 'label'])
y_train = train_df['label']
x_valid = valid_df.drop(columns = ['cust_no', 'label'])
y_valid = valid_df['label']

x_train.fillna(0, inplace=True)
x_valid.fillna(0, inplace=True)

x_train = x_train.values
y_train = y_train.values
x_valid = x_valid.values
y_valid = y_valid.values

# tabnet
start_time = time.time()
clf = TabNetClassifier(n_d = 32, n_a = 32)
clf.fit(
    X_train=x_train,
    y_train=y_train,
    eval_set=[(x_train, y_train),(x_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100, patience=20,
    batch_size=64, virtual_batch_size=128,
)
end_time = time.time()
print('걸린 시간 :', end_time - start_time)

y_pred = clf.predict(x_valid)
print('Accuracy: {:.2f}'.format(accuracy_score(y_valid, y_pred)))
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))




#
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