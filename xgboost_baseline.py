import xgboost as xgb
import pandas as pd
import datetime


time = datetime.datetime.now()
now = time.strftime('%m.%d %H:%M')

train = pd.read_csv('train.csv').replace('False', 0).replace('True', 1)
test = pd.read_csv('test.csv').replace('False', 0).replace('True', 1)
train_drop_base = train.drop(['id', 'contents_open_dt', 'target'], axis=1)
test_drop_base = test.drop(['id', 'contents_open_dt'], axis=1)
train_drop_adv = train_drop_base.drop(['contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                                       'contents_attribute_k'], axis=1)
test_drop_adv = test_drop_base.drop(['contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                                     'contents_attribute_k'], axis=1)

dtrain_1 = xgb.DMatrix(train_drop_base, label=train['target'])
dtest_1 = xgb.DMatrix(test_drop_base)
dtrain_2 = xgb.DMatrix(train_drop_adv, label=train['target'])
dtest_2 = xgb.DMatrix(test_drop_adv)

param = {'gamma': 1, 'objective': 'binary:hinge'}
param['nthread'] = 4
param['eval_metric'] = 'logloss'
num_round = 100

bst_1 = xgb.train(param, dtrain_1, num_round)
pred_1 = bst_1.predict(dtest_1)
bst_2 = xgb.train(param, dtrain_2, num_round)
pred_2 = bst_2.predict(dtest_2)

pred = [0] * len(pred_1)
for i in range(len(pred_1)):
    if pred_1[i] == 1.0 or pred_2[i] == 1.0:
        pred[i] = 1

submission = pd.read_csv('sample_submission.csv')
submission['target'] = pred
submission.to_csv('submissions/baseline_xgboost_{0}.csv'.format(now), index=False)