import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import os, warnings, glob
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
import ml_metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=UserWarning)

# parameters
epochs = 26
input_dim = 4 # Two inputs x1 and x2
learning_rate = 0.005
batch_size = 1
penalty = 5


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim = 8):
        super(LogisticRegression, self).__init__()
        # self.layer1 = torch.nn.Linear(input_dim, 64)
        # self.layer2 = torch.nn.Linear(64, 2)
        self.layer2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        #x = self.layer1(x)
        #y_predicted = torch.sigmoid(self.layer2(y_predicted))
        ouput = self.layer2(x)
        return ouput

class Dataset():
    def __init__(self, path):
        df_train = pd.read_csv(path)
        x = df_train[["predict_label_SFRN","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]].values
        x = (x - x.mean()) / x.std()
        y = df_train.truth_label.values
        for index, label in enumerate(y):
            if int(df_train['predict_label_SFRN'][index]) != int(y[index]):
                y[index] = 1
            else:
                y[index] = 0
        self.x_train = torch.tensor(x, dtype=torch.float)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

# Read in the data and preprocess\
DATASET_FOLDER = "./data/"
path_csv = os.path.join(DATASET_FOLDER, "SP22_test_part_secondhalf.csv")
df_train = pd.read_csv(path_csv)
X = df_train[["predict_label_SFRN","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X= df_train[["predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
y = df_train.truth_label
y_truth = y.copy()
# dataloader
trainset = Dataset(path_csv)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_new.csv"))
X_test = df_test[["predict_label_SFRN","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X_test = df_test[["predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
y_test = df_test.truth_label
y_test_truth = y_test.copy()
path_csv = os.path.join(DATASET_FOLDER, "SP22_result_new.csv")
testset = Dataset(path_csv)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)


model = LogisticRegression(input_dim)
#criterion = torch.nn.BCELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
weights = [1, 1]
class_weights = torch.FloatTensor(weights)
criterion = torch.nn.CrossEntropyLoss(class_weights)
example_result = []
# Training loop
for epoch in range(epochs):
    print('Epoch:', epoch)
    train_iterator = tqdm(train_loader, desc="Train Iteration")
    y_true = list()
    y_pred = list()
    total_loss = 0
    for step, batchs in enumerate(train_iterator):
        batch, targets = batchs[0], batchs[1]
        model.zero_grad()
        pred = model(batch)

        loss = criterion(pred, targets)
        # print("loss {}".format(loss))
        pred_idx = torch.max(pred, 1)[1]
        #pred_idx = torch.round(pred)
        if pred_idx[0] == 1:
            loss = loss * penalty
        # print("pred {}".format(pred_idx))
        # print("loss {}".format(loss))
        loss.backward()
        optimizer.step()
        y_true += list(targets.data.cpu().numpy())
        y_pred += list(pred_idx.data.cpu().numpy())
        # print("y_true {}".format(y_true))
        # print("y_pred {}".format(y_pred))
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    print(
        "Train loss: {} - acc: {}  ".format(total_loss.data.float(), acc))
    if epoch % 5 == 0:
        with torch.no_grad():
            y_true_test = list()
            y_pred_test = list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            for step, batch in enumerate(test_iterator):
                batch, targets = batch[0], batch[1]
                pred = model(batch)
                pred_idx = torch.max(pred, 1)[1]
                #pred_idx = torch.round(pred)
                y_true_test += list(targets.data.cpu().numpy())
                y_pred_test += list(pred_idx.data.cpu().numpy())

            #print(len(y_true_test), len(y_pred_test))
            acc = accuracy_score(y_true_test, y_pred_test)
            #print("Quadratic Weighted Kappa is {}".format(metrics.quadratic_weighted_kappa(y_true, y_pred)))
            print("Test acc is {} ".format(acc))
            # print(classification_report(y_true, y_pred))
            # print(confusion_matrix(y_true, y_pred))
            example_result = y_pred_test

#print(example_result)
deferral_result = example_result
# deferral_result = np.argmax(example_result, axis=1)
unique, counts = np.unique(deferral_result, return_counts=True)
# print(np.asarray((unique, counts)).T)
# np.unique(np.argmax(example_result, axis=1), axis=0)

predict_result = deferral_result
print(deferral_result)

for index, decision in enumerate(deferral_result):
    if decision == 0:
        predict_result[index] = int(df_test['predict_label_SFRN'][index])
    else:
        predict_result[index] = int(y_test_truth[index])

# unique1, counts1 = np.unique(predict_result, return_counts=True)
# print(np.asarray((unique1, counts1)).T)

# unique2, counts2 = np.unique(y_test_truth, return_counts=True)
# print(np.asarray((unique2, counts2)).T)

# print(counts)
# defer_count = 0
# testset_size = counts[0]
print(counts)
# if len(counts) != 1:
defer_count = counts[1]
testset_size = counts[0] + counts[1]
print("Deferral rate is {} ".format(defer_count / testset_size))
acc = accuracy_score(predict_result, y_test_truth)
print("Test acc is {} ".format(acc))
