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

"""

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os, warnings, glob
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=UserWarning)

# Read in the data and preprocess\
DATASET_FOLDER = "./data/"
path_csv = os.path.join(DATASET_FOLDER, "SP22_test_part_secondhalf_sfrn.csv")
df_train = pd.read_csv(path_csv)
X = df_train[["q_id","predict_label_SFRN","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X= df_train[["predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X=df_train[["predict_label_SFRN","predict_label_BERT","prob_incorrect_SFRN","prob_partial_correct_SFRN","prob_correct_SFRN","prob_incorrect_BERT","prob_partial_correct_BERT","prob_correct_BERT"]]
for i in range(len(X)):
    id = X.loc[i, "q_id"]
    X.loc[i, "q_id"] = q_score[id]
    X.loc[i, "q_icc"] = q_icc[id]
    X.loc[i, "q_avesc"] = q_avesc[id]
    X.loc[i, "q_human_acc"] = q_human_acc[id]
y = df_train.truth_label
y_truth = y.copy()

df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_sfrn.csv"))
X_test = df_test[["q_id","predict_label_SFRN","prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X_test = df_test[["predict_label_SFRN", "prob_incorrect_SFRN", "prob_partial_correct_SFRN", "prob_correct_SFRN"]]
#X_test =  df_test[["predict_label_SFRN","predict_label_BERT","prob_incorrect_SFRN","prob_partial_correct_SFRN","prob_correct_SFRN","prob_incorrect_BERT","prob_partial_correct_BERT","prob_correct_BERT"]]
for i in range(len(X_test)):
    id = X_test.loc[i, "q_id"]
    X_test.loc[i, "q_id"] = q_score[id]
    X_test.loc[i, "q_icc"] = q_icc[id]
    X_test.loc[i, "q_avesc"] = q_avesc[id]
    X_test.loc[i, "q_human_acc"] = q_human_acc[id]
y_test = df_test.truth_label
y_test_truth = y_test.copy()

# Create the target labels
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
# for index, label in enumerate(y_test):
#     if (int(df_test['predict_label_SFRN'][index]) != int(y_test[index])) or (
#             int(df_test['predict_label_BERT'][index]) != int(y_test[index])):
#         y_test[index] = 1
#     else:
#         y_test[index] = 0
#
# for index, label in enumerate(y):
#     if (int(df_train['predict_label_SFRN'][index]) != int(y[index])) or (
#             int(df_train['predict_label_BERT'][index]) != int(y[index])):
#         y[index] = 1
#     else:
#         y[index] = 0
# Standardize the input features
X = (X - X.mean()) / X.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Split the training data into a training set and a validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_valid, y_train, y_valid

# Build the model

model = RandomForestClassifier(max_depth=2, random_state=4, n_estimators=4, warm_start=True, class_weight={0: 1, 1: 0.8})
# criterion="entropy",
# Train the model on the training set
model.fit(X_train, y_train)
model.score(X_train, y_train)
print(model.get_params())


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

example_result = model.predict(X_test)
#print(example_result)
for index, decision in enumerate(example_result):
    if decision <= 0:
        example_result[index] = 0
    else:
        example_result[index] = 1
# print(example_result)
deferral_result = example_result
# deferral_result = np.argmax(example_result, axis=1)
unique, counts = np.unique(deferral_result, return_counts=True)
# print(np.asarray((unique, counts)).T)
# np.unique(np.argmax(example_result, axis=1), axis=0)

predict_result = deferral_result.copy()
# print(deferral_result)

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
defer_count, testset_size = 0, len(predict_result)
for i in deferral_result:
    if i == 1: defer_count+=1
print("Deferral rate is {} ".format(defer_count / testset_size))
acc = accuracy_score(predict_result, y_test_truth)
print("Test acc is {} ".format(acc))


# # breakdown by question
# df = pd.read_csv(os.path.join(DATASET_FOLDER, "SP22_result_sfrn.csv"))
# bq_test = df[["a_id", "q_id","predict_label_SFRN", "truth_label"]]
# # bq_test.insert(loc=3,
# #           column='deferral_label',
# #           value=0)
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

