import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch.utils.data
import random
import numpy as np
from tqdm import tqdm
import datetime
import os
import lightgbm as lgb
from embedding import train_modified, test_modified, target
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

time = datetime.datetime.now()
now = time.strftime('%m_%d_%H_%M')
log = 'dnn_800_lightgbm_ensemble_0.45'

random.seed(0)
os.environ["PYTHONHASHSEED"] = str(0)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

EPOCHS = 100
BATCH_SIZE = 512
TRAIN_DNN = False
TRAIN_LGB = False
THRESHOLD = 0.45

# DNN ##################################################################################################################

train = train_modified.resize(train_modified.shape[0], train_modified.shape[1] * train_modified.shape[2])
train_dataset = torch.utils.data.TensorDataset(train, target)
train_num = int(train.shape[0] * 4 / 5)
valid_num = train.shape[0] - train_num
train_set, valid_set = torch.utils.data.random_split(train_dataset, [train_num, valid_num], generator=torch.Generator().manual_seed(42))
valid_target = target[train_num:]

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size = BATCH_SIZE, shuffle = True)

test = test_modified.resize(test_modified.shape[0], test_modified.shape[1] * test_modified.shape[2])
test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = BATCH_SIZE)


class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(500)
        self.batchnorm2 = nn.BatchNorm1d(200)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.batchnorm4 = nn.BatchNorm1d(10)


    def forward(self, inputs):
        inputs = inputs.detach()
        x = self.relu(self.fc1(inputs))
        x = self.batchnorm1(x)
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        # x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.fc4(x))
        x = self.batchnorm4(x)
        x = self.fc5(x)
        return x


input_size = 800
model = DNN(input_size).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)


def train(model, train_loader, optimizer):
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        train_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(data).to(DEVICE)
            targets = targets.unsqueeze(1)
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader.dataset)
        print("train loss : ", train_loss)
    return model

def evaluate(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    output_binary_total = []

    with torch.no_grad():
        for data, targets in valid_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            targets = targets.unsqueeze(1)
            output = model(data)
            loss_func = nn.BCEWithLogitsLoss()
            valid_loss += loss_func(output, targets).item() / len(valid_loader.dataset)
            output_binary = [1 if num >= THRESHOLD else 0 for num in output]
            output_binary_total += output_binary
            answer_compare = [1 if output_binary[i] == targets[i] else 0 for i in range(len(targets))]
            correct += sum(answer_compare)

    accuracy = 100 * correct / len(valid_loader.dataset)
    print('eval loss : {0}, accuracy : {1}'.format(valid_loss, accuracy))

    return output_binary_total, valid_loss, accuracy

def test(model, test_loader):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch).view(-1)
            pred = [1 if pred[i] >= THRESHOLD else 0 for i in range(len(pred))]
            pred_list += pred
    return pred_list


if TRAIN_DNN:
    model = train(model, train_loader, optimizer)
    save_name = 'model/{0}_{1}.pt'.format(now, log)
    torch.save(model.state_dict(), save_name)

else:
    model.load_state_dict(torch.load("model/01_06_15_23_dnn_800_lightgbm_ensemble_0.45.pt"))

valid_pred_dnn, valid_loss, accuracy= evaluate(model, valid_loader)
test_pred_dnn = test(model, test_loader)

for i in range(len(valid_pred_dnn)):
    if valid_pred_dnn[i] < 0.45:
        valid_pred_dnn[i] = 0
    else:
        valid_pred_dnn[i] = 1

for i in range(len(test_pred_dnn)):
    if test_pred_dnn[i] < 0.45:
        test_pred_dnn[i] = 0
    else:
        test_pred_dnn[i] = 1

########################################################################################################################

# lightgbm #############################################################################################################

train = pd.read_csv('train_matched.csv').replace(False, 0).replace(True, 1)
train_drop = train.drop(['id', 'contents_open_dt', 'target'], axis = 1)
test = pd.read_csv('test_matched.csv').replace(False, 0).replace(True, 1)
test_drop = test.drop(['id', 'contents_open_dt'], axis = 1)

param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'binary_logloss'
num_round = 100

target = target.cpu().detach().numpy()

train_data, valid_data, train_target, valid_target = train_test_split(train_drop, target, test_size=0.2, random_state=42)
train_set = lgb.Dataset(train_data, label=train_target, free_raw_data=False)
valid_set = lgb.Dataset(valid_data, label=valid_target, free_raw_data=False)

if TRAIN_LGB:
    bst = lgb.train(param, train_set, num_round, valid_sets=train_set)
    bst.save_model('model/lightgbm_in_ensemble.txt')
else:
    bst = lgb.Booster(model_file='model/lightgbm_in_ensemble.txt')

valid_pred_lgb = bst.predict(valid_data)
test_pred_lgb = bst.predict(test_drop)

for i in range(len(valid_pred_lgb)):
    if valid_pred_lgb[i] < 0.45:
        valid_pred_lgb[i] = 0
    else:
        valid_pred_lgb[i] = 1

for i in range(len(test_pred_lgb)):
    if test_pred_lgb[i] < 0.45:
        test_pred_lgb[i] = 0
    else:
        test_pred_lgb[i] = 1

valid_pred = [0] * len(valid_target)
for i in range(len(valid_pred)):
    if valid_pred_dnn[i] == 1 or valid_pred_lgb[i] == 1:
        valid_pred[i] = 1

f1 = f1_score(valid_target, valid_pred)
tn, fp, fn, tp = confusion_matrix(valid_target, valid_pred).ravel()
print("f1 score : {0}".format(f1))
print("true negative : {0}  | false positive : {1}".format(tn, fp))
print("false negative : {0} | true positive : {1}".format(fn, tp))

test_pred = [0] * (test_drop.shape[0])
for i in range(len(test_pred)):
    if test_pred_dnn[i] == 1 or test_pred_lgb[i] == 1:
        test_pred[i] = 1

valid_data['true'] = valid_target
valid_data['pred'] = valid_pred
valid_data.to_csv('valid_compare_true_and_pred.csv')
exit()

f = open("dnn_feature_dot.txt", 'a')
f.write(log)
f.write("\n")
f.write(now)
f.write("\n")
f.write("f1 score : {0}\n".format(f1))
f.write("true negative : {0}  | false positive : {1}\n".format(tn, fp))
f.write("false negative : {0} | true positive : {1}\n".format(fn, tp))
f.write("\n")
f.close()

submission = pd.read_csv('sample_submission.csv')
submission['target'] = test_pred
submission.to_csv('submissions/{0}_{1}.csv'.format(now, log), index=False)





