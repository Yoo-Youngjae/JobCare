from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import datetime


time = datetime.datetime.now()
now = time.strftime('%m.%d %H:%M')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.drop(['id', 'contents_open_dt','contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                    'contents_attribute_k'], axis=1)
test = test.drop(['id', 'contents_open_dt', 'contents_attribute_i', 'contents_attribute_j_1', 'contents_attribute_j',
                  'contents_attribute_k'], axis=1)

model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1, verbose=2)

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)

preds = model.predict(test)
submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds
submission.to_csv('submissions/baseline_randomf_{0}.csv'.format(now), index=False)