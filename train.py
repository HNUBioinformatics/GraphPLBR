"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : train.py
# @Time : 2022/5/9 11:09
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""


import os
import pickle
from time import *
import pandas as pd
from GCN_Model import *
from sklearn import metrics
from torch.autograd import Variable
from sklearn.model_selection import KFold


Model_Path = './Model/'


def train_one_epoch(model,data_loader):

    epoch_loss_train = 0.0
    n = 0

    for data in data_loader:

        model.optimizer.zero_grad()
        _,_,labels,node_features,graphs = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda())
            graphs = Variable(graphs.cuda())
            y_true = Variable(labels.cuda())
        else:
            node_features = Variable(node_features)
            graphs = Variable(graphs)
            y_true = Variable(labels)

        node_features = torch.squeeze(node_features)
        graphs = torch.squeeze(graphs)
        y_true = torch.squeeze(y_true)

        y_pred = model(node_features, graphs)

        y_true = torch.tensor(y_true, dtype=torch.long)    # calculate loss
        loss = model.criterion(y_pred, y_true)
        loss.backward()     # backward gradient

        model.optimizer.step()     # update all parameters

        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n

    return epoch_loss_train_avg

def evaluate(model,data_loader):

    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = []

    for data in data_loader:

        with torch.no_grad():
            sequence_names,_,labels,node_features,graphs = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                graphs = Variable(graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                y_true = Variable(labels)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            y_true = torch.squeeze(y_true)

            y_pred = model(node_features,graphs)
            y_true = torch.tensor(y_true,dtype = torch.long)
            loss = model.criterion(y_pred,y_true)
            softmax = torch.nn.Softmax(dim = 1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
        epoch_loss_avg = epoch_loss / n

        return epoch_loss_avg,valid_true,valid_pred,pred_dict

def analysis(y_true,y_pred,best_threshold = None):

    if best_threshold == None:

        best_f1 = 0
        best_threshold = 0

        for threshold in range(0,100):

            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true,binary_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    begin_time = time()
    precision = metrics.precision_score(binary_true,binary_pred)
    recall = metrics.recall_score(binary_true,binary_pred)
    mcc = metrics.matthews_corrcoef(binary_true,binary_pred)
    end_time = time()

    run_time = end_time - begin_time
    print('metrics run timr:{}'.format(run_time))

    results = {
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'threshold': best_threshold
    }

    return results

def train(model,train_dataframe,valid_dataframe,fold = 0):

    train_loader = DataLoader(dataset = ProDataset(train_dataframe),batch_size = BATCH_SIZE,shuffle = True,num_workers = 2)
    valid_loader = DataLoader(dataset = ProDataset(valid_dataframe),batch_size = BATCH_SIZE,shuffle = True,num_workers = 2)

    best_epoch = 0
    best_val_recall = 0

    for epoch in range(NUMBER_EPOCHS):

        print('\n========== Train epoch " + str(epoch + 1) + " ==========')
        model.train()

        begin_time = time()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        end_time = time()
        run_time = end_time - begin_time

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Run time:{}".format(run_time))

        record_path = './records/' + 'fold' + str(fold) + '.txt'
        with open(record_path, 'a') as f:
            f.write('epoch' + str(epoch + 1) + '\n')
            f.write("========== Evaluate Valid set ========== \n")
            f.write("Valid loss: {} \n".format(epoch_loss_valid_avg))
            f.write("Valid precision: {} \n".format(result_valid['precision']))
            f.write("Valid recall: {} \n".format(result_valid['recall']))
            f.write("Valid mcc: {} \n".format(result_valid['mcc']))
            f.write("Run time: {} \n".format(run_time))
            f.write('\n')

        if best_val_recall < result_valid['recall']:
            best_epoch = epoch + 1
            best_val_recall = result_valid['recall']
            torch.save(model.state_dict(),os.path.join(Model_Path,'Fold' + str(fold) + '_best_recall_model.pkl'))

    return best_epoch,best_val_recall

def cross_validation(all_dataframe,fold_number = 5):

    print("Random seed:", SEED)
    print("Feature dim:", INPUT_DIM)
    print("Hidden dim:", HIDDEN_DIM)
    print("Layer:", LAYER)
    print("Dropout:", DROPOUT)
    print("Alpha:", ALPHA)
    print("Lambda:", LAMBDA)
    print("Variant:", VARIANT)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)
    print()

    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits = fold_number,shuffle = True)
    fold = 0
    best_epochs = []
    valid_recalls = []

    for train_index,valid_index in kfold.split(sequence_names,sequence_labels):

        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]),
              "samples, validate on",
              str(valid_dataframe.shape[0]),
              "samples")

        model = GraphPLBR(LAYER,INPUT_DIM,HIDDEN_DIM,NUM_CLASSES,DROPOUT,LAMBDA,ALPHA,VARIANT)

        if torch.cuda.is_available():
            model.cuda()

        best_epoch,valid_recall = train(model,train_dataframe,valid_dataframe,fold + 1)
        best_epochs.append(str(best_epoch))
        valid_recalls.append(valid_recall)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average Recall of {} fold：{:.4f}".format(fold_number, sum(valid_recalls) / fold_number))

    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)

def main():

    with open('./data/train_data.pkl', "rb") as f:
        data_all = pickle.load(f)

    IDs, sequences, labels = [], [], []

    for ID in data_all:
        IDs.append(ID)
        item = data_all[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    aver_epoch = cross_validation(train_dataframe, fold_number = 5)


if __name__ == "__main__":
    main()

























