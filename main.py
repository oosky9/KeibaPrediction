from json import load
import numpy as np
import pandas as pd

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

import os
import statistics
import argparse
from tqdm import tqdm

from model import NNwithLSTM, NeuralNetwork, MyDataset

def calc_dice(y_pred, y_true):
    y_pred = y_pred.ge(.5).view(-1).to(torch.float32)
    y_true = y_true.view(-1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    precision = float(precision.cpu().numpy())
    recall = float(recall.cpu().numpy())
    f1 = float(f1.cpu().numpy())

    return precision, recall, f1

def calculate_accuracy(y_pred, y_true):
    predicted = y_pred.ge(.5).view(-1)
    acc = (y_true == predicted).sum().float() / len(y_true)
    return float(acc.cpu().numpy())

def norm(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mu) / std

def norm_max_min(x):
    min = x.min(axis=0, keepdims=True)
    max = x.max(axis=0, keepdims=True)
    result = (x-min)/(max-min)
    return result

def load_data(data, label):
    data = np.asarray(data)

    data = np.asarray(data)
    label  = np.asarray(label)

    data = np.nan_to_num(data)
    label = np.nan_to_num(label)

    data = norm_max_min(data)

    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    dataset = MyDataset(data, label)

    return dataset

def load_data_with_word(data, label, dict):

    x = np.asarray(data[0])
    label  = np.asarray(label)

    x = np.nan_to_num(x)
    label = np.nan_to_num(label)

    x = norm_max_min(x)

    x0 = torch.tensor(x, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    x1 = torch.tensor([dict[0][key] for key in data[1]['name']], dtype=torch.long)
    x2 = torch.tensor([dict[1][key] for key in data[1]['jockey']], dtype=torch.long)

    xs = torch.stack([x1, x2], 1)
    dataset = MyDataset(x0, label, wdata=xs)

    return dataset


def train(args, model, train_data, valid_data):

    writer = SummaryWriter()

    best_dice = 0 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bce_loss = torch.nn.BCELoss()
        

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    result = {}
    result['train/BCE'] = []
    result['train/Precision'] = []
    result['train/Recall'] = []
    result['train/Dice'] = []
    result['train/Accuracy'] = []
    result['valid/BCE'] = []
    result['valid/Precision'] = []
    result['valid/Recall'] = []
    result['valid/Dice'] = []
    result['valid/Accuracy'] = []

    for epoch in range(args.epochs):
        print('train step: epoch {}'.format(str(epoch+1).zfill(4)))

        train_bce = []
        train_pre = []
        train_rec = []
        train_dice = []
        train_acc = []

        model.train()
        for data in tqdm(train_dataloader):
            
            if args.isNLP:
                inp_data = data[0].to(args.device)
                wd_data = data[1].to(args.device)
                lab_data = data[2].to(args.device)
                pred = model(inp_data, wd_data[:, 0], wd_data[:, 1]).squeeze()

            else:
                inp_data = data[0].to(args.device)
                lab_data = data[1].to(args.device)
                pred = model(inp_data).squeeze()
           
            bce = bce_loss(pred, lab_data)
            pre, rec, dice = calc_dice(pred, lab_data)
            acc = calculate_accuracy(pred, lab_data)

            train_bce.append(bce.item())
            train_pre.append(pre)
            train_rec.append(rec)
            train_dice.append(dice)
            train_acc.append(acc)

            model.zero_grad()
            bce.backward()
            optimizer.step()
        
        result['train/BCE'].append(statistics.mean(train_bce))
        result['train/Precision'].append(statistics.mean(train_pre))
        result['train/Recall'].append(statistics.mean(train_rec))
        result['train/Dice'].append(statistics.mean(train_dice))
        result['train/Accuracy'].append(statistics.mean(train_acc))

        writer.add_scalar('train/BinaryCrossEntropy', result['train/BCE'][-1], epoch+1)
        writer.add_scalar('train/Precision', result['train/Precision'][-1], epoch+1)
        writer.add_scalar('train/Recall', result['train/Recall'][-1], epoch+1)
        writer.add_scalar('train/DiceScore', result['train/Dice'][-1], epoch+1)
        writer.add_scalar('train/Accuracy', result['train/Accuracy'][-1], epoch+1)

        print('BCE: {}, Precision: {}, Recall: {}, Dice: {}, Accuracy: {}'.format(result['train/BCE'][-1], result['train/Precision'][-1], result['train/Recall'][-1], result['train/Dice'][-1], result['train/Accuracy'][-1]))

        if (epoch+1) % 10 == 0 or (epoch+1) == 1:

            with torch.no_grad():
                print('valid step: epoch {}'.format(str(epoch+1).zfill(4)))
                model.eval()

                valid_bce = []
                valid_pre = []
                valid_rec = []
                valid_dice = []
                valid_acc = []
                for data in tqdm(valid_dataloader):

                    if args.isNLP:
                        inp_data = data[0].to(args.device)
                        wd_data = data[1].to(args.device)
                        lab_data = data[2].to(args.device)
                        pred = model(inp_data, wd_data[:, 0], wd_data[:, 1]).squeeze()
                    else:
                        inp_data = data[0].to(args.device)
                        lab_data = data[1].to(args.device)
                        pred = model(inp_data).squeeze()

                    bce = bce_loss(pred, lab_data)
                    pre, rec, dice = calc_dice(pred, lab_data)
                    acc = calculate_accuracy(pred, lab_data)

                    valid_bce.append(bce.item())
                    valid_pre.append(pre)
                    valid_rec.append(rec)
                    valid_dice.append(dice)
                    valid_acc.append(acc)
                
                result['valid/BCE'].append(statistics.mean(valid_bce))
                result['valid/Precision'].append(statistics.mean(valid_pre))
                result['valid/Recall'].append(statistics.mean(valid_rec))
                result['valid/Dice'].append(statistics.mean(valid_dice))
                result['valid/Accuracy'].append(statistics.mean(valid_acc))

                writer.add_scalar('valid/BinaryCrossEntropy', result['valid/BCE'][-1], epoch+1)
                writer.add_scalar('valid/Precision', result['valid/Precision'][-1], epoch+1)
                writer.add_scalar('valid/Recall', result['valid/Recall'][-1], epoch+1)
                writer.add_scalar('valid/DiceScore', result['valid/Dice'][-1], epoch+1)
                writer.add_scalar('valid/Accuracy', result['valid/Accuracy'][-1], epoch+1)


                print('BCE: {}, Precision: {}, Recall: {}, Dice: {}, Accuracy: {}'.format(result['valid/BCE'][-1], result['valid/Precision'][-1], result['valid/Recall'][-1], result['valid/Dice'][-1], result['valid/Accuracy'][-1]))


                if best_dice < result['valid/Dice'][-1]:
                    best_dice = result['valid/Dice'][-1]

                    best_model_name = os.path.join(args.save_model_path, f'best_model_{epoch + 1:04}.pth')
                    print('save model ==>> {}'.format(best_model_name))
                    torch.save(model.state_dict(), best_model_name)


def load_dict_list(path):
    import json

    dicts = []
    for p in path:
        with open(p, mode='r') as f:
            data = json.load(f)
        dicts.append(data)
    return dicts

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_model_path', type=str, default='./model/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--isNLP', type=bool, default=True)
    parser.add_argument('--emb_size', type=int, default=10)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--lstm_out', type=int, default=2)

    args = parser.parse_args()
    return args

def main(args):

    check_dir(args.save_model_path)

    df_train = pd.read_csv('./data/train.csv')
    df_valid = pd.read_csv('./data/valid.csv')

    y_train = df_train['answer']
    y_valid = df_valid['answer']

    if args.isNLP:

        dict_list = load_dict_list([
            './data/name_dict.json', 
            './data/jockey_dict.json'
        ])

        x_train = []
        x_valid = []
        x_train.append(df_train[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']])
        x_train.append(df_train[['name', 'jockey']])
        x_valid.append(df_valid[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']])
        x_valid.append(df_valid[['name', 'jockey']])  

        train_data = load_data_with_word(x_train, y_train, dict_list)
        valid_data = load_data_with_word(x_valid, y_valid, dict_list)  

        model = NNwithLSTM(
            in_ch=x_train[0].shape[1], 
            n_hidden=args.n_hidden,
            emb_size=args.emb_size,
            w1_len=len(dict_list[0]),
            w2_len=len(dict_list[1]),
            lstm_hidden=args.lstm_hidden,
            lstm_out_size=args.lstm_out
        )
        model.to(args.device)   

    else:
        x_train = df_train[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']]
        x_valid = df_valid[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']]
        
        train_data = load_data(x_train, y_train)
        valid_data = load_data(x_valid, y_valid)

        model = NeuralNetwork(in_ch=x_train.shape[1], n_hidden=args.n_hidden)
        model.to(args.device)

    train(args, model, train_data, valid_data)


if __name__ == '__main__':

    args = arg_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
