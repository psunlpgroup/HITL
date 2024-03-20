"""
Exp #1
X=["prob_incorrect_SFRN’’,"prob_partial_correct_SFRN","prob_correct_SFRN"]
Exp #2
X=["predict_label_SFRN","prob_incorrect_SFRN’’,"prob_partial_correct_SFRN","prob_correct_SFRN"]
Exp #3
X=["Quesetion_id","prob_incorrect_SFRN’’,"prob_partial_correct_SFRN","prob_correct_SFRN"]
Exp #4
X=["predict_label_SFRN", “next_high_class”,"prob_incorrect_SFRN’’,"prob_partial_correct_SFRN","prob_correct_SFRN"]
Exp #5
X=["Quesetion_id", "predict_label_SFRN", “next highest class”, "prob_incorrect_SFRN’’,"prob_partial_correct_SFRN","prob_correct_SFRN"]


BERT + SFRN

X=["q_id","predict_label_SFRN","next_high_class_SFRN","predict_label_BERT","next_high_class_BERT","prob_incorrect_SFRN","prob_partial_correct_SFRN","prob_correct_SFRN","prob_incorrect_BERT","prob_partial_correct_BERT","prob_correct_BERT"]
"""


#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os, warnings, glob
import pandas as pd
import numpy as np
import random
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=UserWarning)

q_avesc = {
0: 1.88,
1: 1.22,
2: 1.12,
3: 1.07,
4: 1.43,
5: 1.85,
6: 1.72,
7: 1.56,
8: 1.31
}

q_human_acc = {
0: 0.97,
1: 0.95,
2: 0.92,
3: 0.91,
4: 0.89,
5: 0.96,
6: 0.94,
7: 0.93,
8: 0.91
}

q_icc = {
0: 0.93,
1: 0.92,
2: 0.92,
3: 0.92,
4: 0.84,
5: 0.95,
6: 0.94,
7: 0.91,
8: 0.87
}

q_score = {
0: 0.9318,
1: 0.9574,
2: 0.9361,
3: 0.8367,
4: 0.5306,
5: 0.825,
6: 0.775,
7: 0.7,
8: 0.9340
}

q_score_2 = {
0: 0.8928,
1: 0.7962,
2: 0.6269,
3: 0.7551,
4: 0.58,
5: 0.6923,
6: 0.78,
7: 0.8461,
8: 0.96
}

# Read in the data and preprocess\
DATASET_FOLDER = "./data/"
path_csv = os.path.join(DATASET_FOLDER, "SP22_test_part_secondhalf_sfrn.csv")
df_train = pd.read_csv(path_csv)
#X = df_train[["q_id","predict_label_SFRN","next_high_class","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
X = df_train[[ "q_id","predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X=df_train[["predict_label_SFRN","predict_label_BERT","prob_incorrect_SFRN","prob_partial_correct_SFRN","prob_correct_SFRN","prob_incorrect_BERT","prob_partial_correct_BERT","prob_correct_BERT"]]
# X.insert(loc=1,
#           column='q_score',
#           value=0.0)
for i in range(len(X)):
    id = X.loc[i, "q_id"]
    X.loc[i, "q_id"] = q_score[id]
    if X.loc[i, "predict_label_SFRN"] == 0:
        X.loc[i, "predict_label_SFRN"] = 0.45
    if X.loc[i, "predict_label_SFRN"] == 1:
        X.loc[i, "predict_label_SFRN"] = 0.35
    if X.loc[i, "predict_label_SFRN"] == 2:
        X.loc[i, "predict_label_SFRN"] = 0.2
    #X.loc[i, "q_human_acc"] = random.gauss(mu=q_score_2[id], sigma=1)
    #X.loc[i, "q_human_acc"] = random.uniform(0, 1) * q_human_acc[id]

y = df_train.truth_label
y_truth = y.copy()


df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_sfrn.csv"))
#X_test = df_test[["q_id","predict_label_SFRN","next_high_class","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
X_test = df_test[["q_id","predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X_test =  df_test[["predict_label_SFRN","predict_label_BERT","prob_incorrect_SFRN","prob_partial_correct_SFRN","prob_correct_SFRN","prob_incorrect_BERT","prob_partial_correct_BERT","prob_correct_BERT"]]
# X_test.insert(loc=1,
#           column='q_score',
#           value=0.0)
for i in range(len(X_test)):
    id = X_test.loc[i, "q_id"]
    X_test.loc[i, "q_id"] = q_score[id]
    if X_test.loc[i, "predict_label_SFRN"] == 0:
        X_test.loc[i, "predict_label_SFRN"] = 0.45
    if X_test.loc[i, "predict_label_SFRN"] == 1:
        X_test.loc[i, "predict_label_SFRN"] = 0.35
    if X_test.loc[i, "predict_label_SFRN"] == 2:
        X_test.loc[i, "predict_label_SFRN"] = 0.2
    #X_test.loc[i, "q_human_acc"] = random.gauss(mu=q_score_2[id], sigma=1)
    #X_test.loc[i, "q_human_acc"] = random.uniform(0, 1)

y_test = df_test.truth_label
y_test_truth = y_test.copy()

# Create the target labels
# for index,label in enumerate(y_test):
#     if (int(df_test['predict_label_SFRN'][index])!=int(y_test[index])) or (int(df_test['predict_label_BERT'][index])!=int(y_test[index])):
#         y_test[index]=1
#     else: y_test[index]=0
#
# for index,label in enumerate(y):
#     if (int(df_train['predict_label_SFRN'][index])!=int(y[index])) or (int(df_train['predict_label_BERT'][index])!=int(y[index])):
#         y[index]=1
#     else: y[index]=0

for index, label in enumerate(y_test):
    if int(df_test['predict_label_SFRN'][index]) != int(y_test[index]):
        y_test[index] = 1
    else:
        y_test[index] = 0

for index, label in enumerate(y):
    if int(df_train['predict_label_SFRN'][index]) != int(y[index]):
        y[index] = 1
    else:
        y[index] = 0

# Standardize the input features
# X = (X - X.mean()) / X.std()
# X_test = (X_test - X_test.mean()) / X_test.std()
# Split the training data into a training set and a validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_valid, y_train, y_valid

# Build the model
model = LogisticRegression(penalty='none', C=20,  class_weight={0: 1, 1: 1}) #, penalty='none' , tol=0.0001 solver="saga",
#model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)
model.score(X_train, y_train)
print(model.coef_)
print(model.intercept_)

# Evaluate the model on the validation set
score = model.score(X_valid, y_valid)
print("Validation set score: {:.3f}".format(score))

# Evaluate the model on the test set
score = model.score(X_test, y_test)
print("Test set score: {:.3f}".format(score))

# # Use the trained model to make predictions on new, unseen data
# new_data = [[0.5, 0.2, 0.3]] # example of new data with the same format as X
# prediction = model.predict(new_data)
# print("Prediction for new data: {}".format(prediction))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#### Dev

example_result = model.predict(X)
deferral_result = example_result
unique, counts = np.unique(deferral_result, return_counts=True)
predict_result = deferral_result.copy()
for index, decision in enumerate(deferral_result):
    if decision == 0:
        predict_result[index] = int(df_train['predict_label_SFRN'][index])
    else:
        predict_result[index] = int(y_truth[index])
print(counts)
defer_count, devset_size = 0, len(predict_result)
for i in deferral_result:
    if i == 1: defer_count+=1
print("Dev Deferral rate is {} ".format(defer_count / devset_size))
acc = accuracy_score(predict_result, y_truth)
print("Dev acc is {} ".format(acc))
# Get CI
rng = np.random.RandomState(seed=12345)
idx = np.arange(y.shape[0])
print(y.shape[0]*0.9)
test_accuracies = []
test_df = []
for i in range(200):
    pred_idx = rng.choice(idx, size=420, replace=False)
    acc_test_boot = np.mean(predict_result[pred_idx] == y_truth[pred_idx])
    unique, counts = np.unique(deferral_result[pred_idx], return_counts=True)
    defer_count, devset_size = 0, len(predict_result)
    for i in deferral_result:
        if i == 1: defer_count += 1
    test_df.append(defer_count / devset_size)
    #print("Test Deferral rate is {} ".format(defer_count / testset_size))
    test_accuracies.append(acc_test_boot)

bootstrap_mean = np.mean(test_accuracies)
print("Dev Accuracy mean {}".format(bootstrap_mean))
ci_lower = np.percentile(test_accuracies, 2.5)
ci_upper = np.percentile(test_accuracies, 97.5)
print("Dev Accuracy [ {} , {}]".format(ci_lower, ci_upper))

bootstrap_mean = np.mean(test_df)
print("Dev Df mean {}".format(bootstrap_mean))
ci_lower = np.percentile(test_df, 2.5)
ci_upper = np.percentile(test_df, 97.5)
print("Dev Df [ {} , {}]".format(ci_lower, ci_upper))


#### Test

example_result = model.predict(X_test)

deferral_result = example_result
unique, counts = np.unique(deferral_result, return_counts=True)

predict_result = deferral_result.copy()
# print(deferral_result)

for index,decision in enumerate(deferral_result): 
    if decision==0:
        predict_result[index]=int(df_test['predict_label_SFRN'][index])
    else: predict_result[index]=int(y_test_truth[index])

print(counts)
print(counts)
defer_count, testset_size = 0, len(predict_result)
for i in deferral_result:
    if i == 1: defer_count+=1
print("Test Deferral rate is {} ".format(defer_count / testset_size))
#print(predict_result)
acc = accuracy_score(predict_result,y_test_truth)
print("Test acc is {} ".format(acc))


# Get bootstrap
print(model.get_params())

rng = np.random.RandomState(seed=12345)
idx = np.arange(y_test.shape[0])
print(y_test.shape[0]*0.9)
test_accuracies = []
test_df = []

for i in range(200):

    pred_idx = rng.choice(idx, size=420, replace=False)
    acc_test_boot = np.mean(predict_result[pred_idx] == y_test_truth[pred_idx])
    # print(predict_result[pred_idx])
    # print(y_test_truth[pred_idx])
    unique, counts = np.unique(deferral_result[pred_idx], return_counts=True)
    defer_count, testset_size = 0, len(predict_result)
    for i in deferral_result:
        if i == 1: defer_count += 1
    test_df.append(defer_count / testset_size)
    #print("Test Deferral rate is {} ".format(defer_count / testset_size))
    test_accuracies.append(acc_test_boot)

bootstrap_mean = np.mean(test_accuracies)
print("Test Accuracy mean {}".format(bootstrap_mean))
ci_lower = np.percentile(test_accuracies, 2.5)
ci_upper = np.percentile(test_accuracies, 97.5)
print("Test Accuracy [ {} , {}]".format(ci_lower, ci_upper))

bootstrap_mean = np.mean(test_df)
print("Test Df mean {}".format(bootstrap_mean))
ci_lower = np.percentile(test_df, 2.5)
ci_upper = np.percentile(test_df, 97.5)
print("Test Df [ {} , {}]".format(ci_lower, ci_upper))

# output result
# df = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_sfrn.csv"))
# #X_test = df_test[["q_id","predict_label_SFRN","next_high_class","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
# out = df[["a_id", "q_id", "predict_label_SFRN", "truth_label"]]
# d = {"a_id":[], "q_id":[], "predict_label":[], "truth_label":[]}
# out_new = pd.DataFrame(data=d)
# for i in range(len(out)):
#     id = out.loc[i, "a_id"]
#     split = id.split("_")
#     out_new.loc[i, "a_id"] = split[0]
#     out_new.loc[i, "predict_label"] = str(predict_result[i])
#     out_new.loc[i, "truth_label"] = str(out.loc[i, "truth_label"])
#     out_new.loc[i, "q_id"] = split[1]+"_"+split[2]
#     #print(out.loc[i])
#
# out_new.to_csv("./results/model_prediction_4.csv", index=False)
#
#
# breakdown by question
# df = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_sfrn.csv"))
# bq_test = df[["a_id", "q_id","predict_label_SFRN", "truth_label"]]
# bq_test.insert(loc=3,
#           column='deferral_label',
#           value=0)
# for i in range(len(bq_test)):
#     id = bq_test.loc[i, "a_id"]
#     split = id.split("_")
#     bq_test.loc[i, "a_id"] = split[0]
#     bq_test.loc[i, "deferral_label"] = deferral_result[i]
#     bq_test.loc[i, "predict_label_after_defer"] = predict_result[i]
#     bq_test.loc[i, "q_id"] = split[1]+"_"+split[2]
# #print(bq_test)
# gb = bq_test.groupby('q_id')
#
# for q in gb.groups:
#     print("##################"+q+"#################")
#     q_gp = gb.get_group(q)
#     predict_result = list(q_gp.predict_label_after_defer.to_dict().values())
#     deferral_result = list(q_gp.deferral_label.to_dict().values())
#     truth_label = list(q_gp.truth_label.to_dict().values())
#     unique, counts = np.unique(deferral_result, return_counts=True)
#     print(counts)
#     defer_count, testset_size = 0, len(predict_result)
#     for i in deferral_result:
#         if i == 1: defer_count+=1
#     print("Test Deferral rate of {} is {} ".format(q,(defer_count / testset_size)))
#     #print(predict_result)
#     acc = accuracy_score(predict_result, truth_label)
#     print("Test acc of {} is {} ".format(q, acc))



# Get CI from files
#
# df_test = pd.read_csv("./results/SP22_result_SFRN_test_manualpolicy.csv")
# #X_test = df_test[["q_id","predict_label_SFRN","next_high_class","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
# x = df_test[[ "predict_label", "truth_label", "defer_decision"]]
#
# # y_truth = list(x.truth_label.to_dict().values())
# # predict_result = list(x.predict_label.to_dict().values())
# y_truth = x.truth_label
# predict_result = x.predict_label
# deferral_result = x.defer_decision
# # for i in range(len(deferral_result)):
# #     if y_truth.loc[i] == predict_result[]
# #     id = bq_test.loc[i, "a_id"]
#
# rng = np.random.RandomState(seed=12345)
# idx = np.arange(y.shape[0])
# print(y.shape[0]*0.9)
# test_accuracies = []
# test_df = []
#
# for i in range(200):
#     pred_idx = rng.choice(idx, size=420, replace=False)
#     acc_test_boot = np.mean(predict_result[pred_idx] == y_truth[pred_idx])
#     unique, counts = np.unique(deferral_result[pred_idx], return_counts=True)
#     defer_count = counts[1]
#     testset_size = counts[0] + counts[1]
#     test_df.append(defer_count / testset_size)
#     print("Test Deferral rate is {} ".format(defer_count / testset_size))
#     test_accuracies.append(acc_test_boot)
#
# bootstrap_mean = np.mean(test_accuracies)
# print("Accuracy mean {}".format(bootstrap_mean))
# ci_lower = np.percentile(test_accuracies, 2.5)
# ci_upper = np.percentile(test_accuracies, 97.5)
# print("Accuracy [ {} , {}]".format(ci_lower, ci_upper))
#
# bootstrap_mean = np.mean(test_df)
# print("Df mean {}".format(bootstrap_mean))
# ci_lower = np.percentile(test_df, 2.5)
# ci_upper = np.percentile(test_df, 97.5)
# print("Df [ {} , {}]".format(ci_lower, ci_upper))
