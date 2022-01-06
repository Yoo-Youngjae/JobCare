import lightgbm as lgb
import pandas as pd
import datetime
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from embedding import train_modified, test_modified, target
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

time = datetime.datetime.now()
now = time.strftime('%m_%d_%H_%M')
log = 'lightgbm_with_matched_0.45'
TRAIN = True

train = pd.read_csv('train_matched.csv').replace(False, 0).replace(True, 1)
train_drop = train.drop(['id', 'contents_open_dt', 'target'], axis = 1)
test = pd.read_csv('test_matched.csv').replace(False, 0).replace(True, 1)
test_drop = test.drop(['id', 'contents_open_dt'], axis = 1)

param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'binary_logloss'
num_round = 100

# train_modified = train_modified.cpu().detach().numpy()
# train_modified.resize(train_modified.shape[0], train_modified.shape[1] * train_modified.shape[2])
# test_modified = test_modified.cpu().detach().numpy()
# test_modified.resize(test_modified.shape[0], test_modified.shape[1] * test_modified.shape[2])
target = target.cpu().detach().numpy()

train_data, valid_data, train_target, valid_target = train_test_split(train_drop, target, test_size=0.2, random_state=42)
train_set = lgb.Dataset(train_data, label=train_target, free_raw_data=False)
valid_set = lgb.Dataset(valid_data, label=valid_target, free_raw_data=False)

if TRAIN:
    bst = lgb.train(param, train_set, num_round, valid_sets=train_set)
    bst.save_model('model/lightgbm_baseline.txt')
else:
    bst = lgb.Booster(model_file='model/lightgbm_baseline.txt')

pred_valid = bst.predict(valid_data)
for i in range(len(pred_valid)):
    if pred_valid[i] < 0.45:
        pred_valid[i] = 0
    else:
        pred_valid[i] = 1

f1 = f1_score(valid_target, pred_valid)
tn, fp, fn, tp = confusion_matrix(valid_target, pred_valid).ravel()
print("f1 score : {0}".format(f1))
print("true negative : {0}  | false positive : {1}".format(tn, fp))
print("false negative : {0} | true positive : {1}".format(fn, tp))

# valid_data['true'] = valid_target
# valid_data['pred'] = pred_valid
# valid_data.to_csv('valid_compare_true_and_pred.csv')

pred_test = bst.predict(test_drop)
for i in range(len(pred_test)):
    if pred_test[i] < 0.45:
        pred_test[i] = 0
    else:
        pred_test[i] = 1
submission = pd.read_csv('sample_submission.csv')
submission['target'] = pred_test
submission.to_csv('submissions/{0}_{1}.csv'.format(now, log), index=False)


