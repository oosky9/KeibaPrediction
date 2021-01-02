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

from model import NeuralNetwork, MyDataset

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
    return float(f1.cpu().numpy())

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


def train(args, x_train, y_train, x_valid, y_valid):

    writer = SummaryWriter()

    best_dice = 0 

    model = NeuralNetwork(in_ch=x_train.shape[1], n_hidden=args.n_hidden)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bce_loss = torch.nn.BCELoss()

    train_data = load_data(x_train, y_train)
    valid_data = load_data(x_valid, y_valid)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    result = {}
    result['train/BCE'] = []
    result['train/Dice'] = []
    result['train/Accuracy'] = []
    result['valid/BCE'] = []
    result['valid/Dice'] = []
    result['valid/Accuracy'] = []

    for epoch in range(args.epochs):
        print('train step: epoch {}'.format(str(epoch+1).zfill(4)))

        train_bce = []
        train_dice = []
        train_acc = []

        for inp_data, lab_data in tqdm(train_dataloader):
            inp_data = inp_data.to(args.device)
            lab_data = lab_data.to(args.device)

            pred = model(inp_data)

            bce = bce_loss(pred, lab_data)
            dice = calc_dice(pred, lab_data)
            acc = calculate_accuracy(pred, lab_data)

            train_bce.append(bce.item())
            train_dice.append(dice)
            train_acc.append(acc)

            model.zero_grad()
            bce.backward()
            optimizer.step()
        
        result['train/BCE'].append(statistics.mean(train_bce))
        result['train/Dice'].append(statistics.mean(train_dice))
        result['train/Accuracy'].append(statistics.mean(train_acc))

        writer.add_scalar('train/BinaryCrossEntropy', result['train/BCE'][-1], epoch+1)
        writer.add_scalar('train/DiceScore', result['train/Dice'][-1], epoch+1)
        writer.add_scalar('train/Accuracy', result['train/Accuracy'][-1], epoch+1)

        print('BCE: {}, Dice: {}, Accuracy: {}'.format(result['train/BCE'][-1], result['train/Dice'][-1], result['train/Accuracy'][-1]))

        if (epoch+1) % 10 == 0 or (epoch+1) == 1:

            with torch.no_grad():
                print('valid step: epoch {}'.format(str(epoch+1).zfill(4)))
                model.eval()

                valid_bce = []
                valid_dice = []
                valid_acc = []
                for inp_data, lab_data in tqdm(valid_dataloader):
                    inp_data = inp_data.to(args.device)
                    lab_data = lab_data.to(args.device)

                    pred = model(inp_data)

                    bce = bce_loss(pred, lab_data)
                    dice = calc_dice(pred, lab_data)
                    acc = calculate_accuracy(pred, lab_data)

                    valid_bce.append(bce.item())
                    valid_dice.append(dice)
                    valid_acc.append(acc)
                
                result['valid/BCE'].append(statistics.mean(valid_bce))
                result['valid/Dice'].append(statistics.mean(valid_dice))
                result['valid/Accuracy'].append(statistics.mean(valid_acc))

                writer.add_scalar('valid/BinaryCrossEntropy', result['valid/BCE'][-1], epoch+1)
                writer.add_scalar('valid/DiceScore', result['valid/Dice'][-1], epoch+1)
                writer.add_scalar('valid/Accuracy', result['valid/Accuracy'][-1], epoch+1)


                print('BCE: {}, Dice: {}, Accuracy: {}'.format(result['valid/BCE'][-1], result['valid/Dice'][-1], result['valid/Accuracy'][-1]))


                if best_dice < result['valid/Dice'][-1]:
                    best_dice = result['valid/Dice'][-1]

                    best_model_name = os.path.join(args.save_model_path, f'best_model_{epoch + 1:04}.pth')
                    print('save model ==>> {}'.format(best_model_name))
                    torch.save(model.state_dict(), best_model_name)


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

    args = parser.parse_args()
    return args

def main(args):

    check_dir(args.save_model_path)

    df_train = pd.read_csv('./data/train.csv')
    df_valid = pd.read_csv('./data/valid.csv')

    x_train = df_train[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']]
    
    x_valid = df_valid[['code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']]

    y_train = df_train['answer']
    y_valid = df_valid['answer']


    train(args, x_train, y_train, x_valid, y_valid)


if __name__ == '__main__':

    args = arg_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
