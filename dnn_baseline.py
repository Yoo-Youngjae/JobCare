import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
import random
import numpy as np
from tqdm import tqdm
import datetime
import os

time = datetime.datetime.now()
now = time.strftime('%m.%d %H:%M')
log = 'dnn_baseline_epoch_50_th_0.45'

random.seed(0)
os.environ["PYTHONHASHSEED"] = str(0)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if min[c] == max[c]:
            ndf[c] = df[c] - min[c]
        else:
            ndf[c] = (df[c] - min[c]) / (max[c] - min[c])
    return ndf

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

EPOCHS = 5
BATCH_SIZE = 216
TRAIN = True

train_df = pd.read_csv('train.csv').replace(False, 0).replace(True, 1)
test_df = pd.read_csv('test.csv').replace(False, 0).replace(True, 1)
train_target = torch.tensor(train_df['target'].values.astype(np.float32))

train_drop_base = train_df.drop(['id', 'contents_open_dt', 'target'], axis = 1)
train_drop_adv = train_df.drop(['id', 'contents_open_dt', 'target', 'contents_attribute_i',
                                'contents_attribute_j_1', 'contents_attribute_j','contents_attribute_k'], axis = 1)
test_drop_base = test_df.drop(['id', 'contents_open_dt'], axis = 1)
test_drop_adv = test_df.drop(['id', 'contents_open_dt', 'contents_attribute_i',
                                'contents_attribute_j_1', 'contents_attribute_j','contents_attribute_k'], axis = 1)
total_base = pd.concat([train_drop_base, test_drop_base])
total_adv = pd.concat([train_drop_adv, test_drop_adv])

# min = total_base.min()
# max = total_base.max()
# train_drop_base = normalize(train_drop_base)
# train_drop_adv = normalize(train_drop_adv)
# test_drop_base = normalize(test_drop_base)
# test_drop_adv = normalize(test_drop_adv)

train_tensor_1 = torch.tensor(train_drop_base.values.astype(np.float32))
train_dataset_1 = data.TensorDataset(train_tensor_1, train_target)
train_loader_1 = data.DataLoader(dataset = train_dataset_1, batch_size = BATCH_SIZE, shuffle = True)

train_tensor_2 = torch.tensor(train_drop_adv.values.astype(np.float32))
train_dataset_2 = data.TensorDataset(train_tensor_2, train_target)
train_loader_2 = data.DataLoader(dataset = train_dataset_2, batch_size = BATCH_SIZE, shuffle = True)

test_tensor_1= torch.tensor(test_drop_base.values.astype(np.float32))
test_loader_1 = data.DataLoader(dataset = test_tensor_1, batch_size = BATCH_SIZE)

test_tensor_2 = torch.tensor(test_drop_adv.values.astype(np.float32))
test_loader_2 = data.DataLoader(dataset = test_tensor_2, batch_size = BATCH_SIZE)


class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(8)


    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.batchnorm1(x)
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        # x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.fc4(x)
        return x


input_size_1 = train_df.shape[1] - 3
model_1 = DNN(input_size_1).to(DEVICE)
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.001)

input_size_2 = train_df.shape[1] - 7
model_2  = DNN(input_size_2).to(DEVICE)
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.001)

def train(model, train_loader, optimizer):
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        avg_loss = 0
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            target = target.unsqueeze(1)
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        print("train loss : ", avg_loss)
    return model

def test(model, test_loader):
    model.eval()
    ypred_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            ypred = model(batch).view(-1)
            ypred = [1 if ypred[i] >= 0.45 else 0 for i in range(len(ypred))]
            ypred_list += ypred
    return ypred_list

if TRAIN:
    model_1 = train(model_1, train_loader_1, optimizer_1)
    # model_2 = train(model_2, train_loader_2, optimizer_2)
    save_name_1 = '../model_1/{0}_{1}.pt'.format(log, now)
    save_name_2 = '../model_2/{0}_{1}.pt'.format(log, now)
    torch.save(model_1.state_dict(), save_name_1)
    # torch.save(model_2.state_dict(), save_name_2)
else:
    model_1.load_state_dict(torch.load("../model_1/dnn_baseline_epoch_50_th_0.45_12.29 02:01.pt"))
    # model_2.load_state_dict(torch.load("../model_2/dnn_baseline_epoch_50_th_0.45_12.28 20:46.pt"))


ypred_1 = test(model_1, test_loader_1)
# ypred_2 = test(model_2, test_loader_2)
#
# pred = [0] * len(ypred_1)
# for i in range(len(ypred_1)):
#     if ypred_1[i] == 1 or ypred_2[i] == 1:
#         pred[i] = 1

submission = pd.read_csv('sample_submission.csv')
submission['target'] = ypred_1
submission.to_csv('../submissions/{0}_{1}.csv'.format(log, now), index=False)



