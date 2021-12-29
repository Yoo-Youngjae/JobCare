import lightgbm as lgb
import pandas as pd
import datetime

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

time = datetime.datetime.now()
now = time.strftime('%m.%d %H:%M')

train = pd.read_csv('train.csv').replace(False, 0).replace(True, 1)
test = pd.read_csv('test.csv').replace(False, 0).replace(True, 1)

train_drop_base = train.drop(['id', 'contents_open_dt', 'target'], axis=1)
test_drop_base = test.drop(['id', 'contents_open_dt'], axis=1)
TAG_MIN = train_drop_base.min()
TAG_MAX = train_drop_base.max()
train_drop_base = normalize(train_drop_base).ewm(alpha=0.9).mean()
test_drop_base = normalize(test_drop_base).ewm(alpha=0.9).mean()


train_drop_adv = train_drop_base.drop(['contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                                       'contents_attribute_k'], axis=1)
test_drop_adv = test_drop_base.drop(['contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                                      'contents_attribute_k'], axis=1)
TAG_MIN = train_drop_adv.min()
TAG_MAX = train_drop_adv.max()
train_drop_adv = normalize(train_drop_adv).ewm(alpha=0.9).mean()
test_drop_adv = normalize(test_drop_adv).ewm(alpha=0.9).mean()


train_data_1 = lgb.Dataset(train_drop_base, label=train['target'], free_raw_data=False)
train_data_2 = lgb.Dataset(train_drop_adv, label=train['target'], free_raw_data=False)
param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'binary_logloss'

num_round = 100
bst_1 = lgb.train(param, train_data_1, num_round, valid_sets=train_data_1)
ypred_1 = bst_1.predict(test_drop_base)
bst_2 = lgb.train(param, train_data_2, num_round, valid_sets=train_data_2)
ypred_2 = bst_2.predict(test_drop_adv)

for i in range(len(ypred_1)):
    if ypred_1[i] < 0.45:
        ypred_1[i] = 0
    else:
        ypred_1[i] = 1
    if ypred_2[i] < 0.45:
        ypred_2[i] = 0
    else:
        ypred_2[i] = 1

pred = [0] * len(ypred_1)
for i in range(len(ypred_1)):
    if ypred_1[i] == 1.0 or ypred_2[i] == 1.0:
        pred[i] = 1

submission = pd.read_csv('sample_submission.csv')
submission['target'] = pred
submission.to_csv('submissions/baseline_lightgbm_{0}.csv'.format(now), index=False)


