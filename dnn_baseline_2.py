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
from embedding import train_modified, test_modified, target
from sklearn.metrics import f1_score, confusion_matrix

time = datetime.datetime.now()
now = time.strftime('%m_%d_%H_%M')
log = 'dnn_baseline_800_0.45'

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
TRAIN = False
THRESHOLD = 0.45
IS_BASIC = True

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


class DNN_feature_dot(nn.Module):
    def __init__(self, input_size_m, input_size_p, input_size_c):
        super(DNN_feature_dot, self).__init__()
        self.input_size_m = input_size_m
        self.input_size_p = input_size_p
        self.input_size_c = input_size_c

        self.fc_m_1 = nn.Linear(input_size_m, 500)
        self.fc_p_1 = nn.Linear(input_size_p, 500)
        self.fc_c_1 = nn.Linear(input_size_c, 500)

        self.fc_2 = nn.Linear(500, 100)
        self.fc_3 = nn.Linear(100, 10)

        self.fc = nn.Linear(30, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(500)
        self.batchnorm2 = nn.BatchNorm1d(100)

    def forward(self, inputs):
        match = inputs[:, : self.input_size_m]
        match = match.detach()
        person = inputs[:, self.input_size_m : self.input_size_m + self.input_size_p]
        person = person.detach()
        contents = inputs[:, self.input_size_m + self.input_size_p :  self.input_size_m + self.input_size_p + self.input_size_c]
        contents = contents.detach()

        m = self.relu(self.fc_m_1(match))
        m = self.relu(self.fc_2(m))
        m = self.relu(self.fc_3(m))

        p = self.relu(self.fc_p_1(person))
        p = self.relu(self.fc_2(p))
        p = self.relu(self.fc_3(p))

        c = self.relu(self.fc_c_1(contents))
        c = self.relu(self.fc_2(c))
        c = self.relu(self.fc_3(c))

        dot_1 = torch.sum(m * p, dim=-1)
        dot_2 = torch.sum(p * c, dim=-1)
        dot_3 = torch.sum(c * m, dim=-1)

        dot = torch.stack((dot_1, dot_2, dot_3))
        dot = dot.transpose(0, 1)
        dot = self.fc(dot)

        return dot


class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(3000)
        self.batchnorm2 = nn.BatchNorm1d(1000)
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

input_size_m = 120
input_size_p = 260
input_size_c = 220
input_size = 800

if IS_BASIC:
    model = DNN(input_size).to(DEVICE)
else:
    model = DNN_feature_dot(input_size_m, input_size_p, input_size_c).to(DEVICE)
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

if TRAIN:
    model = train(model, train_loader, optimizer)
    save_name = 'model/{0}_{1}.pt'.format(now, log)
    torch.save(model.state_dict(), save_name)

else:
    model.load_state_dict(torch.load("model/01_06_14_40_dnn_baseline_800_0.45.pt"))

valid_pred, valid_loss, accuracy= evaluate(model, valid_loader)
f1 = f1_score(valid_target, valid_pred)
tn, fp, fn, tp = confusion_matrix(valid_target, valid_pred).ravel()
print("f1 score : {0}".format(f1))
print("true negative : {0}  | false positive : {1}".format(tn, fp))
print("false negative : {0} | true positive : {1}".format(fn, tp))

if IS_BASIC:
    f = open("dnn.txt", 'a')
    f.write(log)
    f.write("\n")
    f.write(now)
    f.write("\n")
    f.write('eval loss : {0}, accuracy : {1}\n'.format(valid_loss, accuracy))
    f.write("f1 score : {0}\n".format(f1))
    f.write("true negative : {0}  | false positive : {1}\n".format(tn, fp))
    f.write("false negative : {0} | true positive : {1}\n".format(fn, tp))
    f.write("\n")
    f.close()
else:
    f = open("dnn_feature_dot.txt", 'a')
    f.write(log)
    f.write("\n")
    f.write(now)
    f.write("\n")
    f.write('eval loss : {0}, accuracy : {1}\n'.format(valid_loss, accuracy))
    f.write("f1 score : {0}\n".format(f1))
    f.write("true negative : {0}  | false positive : {1}\n".format(tn, fp))
    f.write("false negative : {0} | true positive : {1}\n".format(fn, tp))
    f.write("\n")
    f.close()

ypred = test(model, test_loader)
submission = pd.read_csv('sample_submission.csv')
submission['target'] = ypred
submission.to_csv('submissions/{0}_{1}.csv'.format(now, log), index=False)

