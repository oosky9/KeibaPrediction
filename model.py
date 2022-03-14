import torch
from torch import nn

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, wdata=None, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label
        self.wdata = wdata

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data)[0][idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]

        if self.wdata is None:
            return out_data, out_label
        else:
            out_wdata = self.wdata[idx]
            return out_data, out_wdata, out_label 


class NeuralNetwork(nn.Module):

    def __init__(self, in_ch, n_hidden):
        super().__init__()

        self.fc1 = nn.Linear(in_ch, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden*2)
        self.fc3 = nn.Linear(n_hidden*2, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 1)

        self.drop = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.drop(out)
        out = self.sigmoid(self.fc4(out))

        return out



class NNwithLSTM(nn.Module):

    def __init__(self, in_ch, n_hidden, emb_size, w1_len, w2_len, lstm_hidden, lstm_out_size):
        super().__init__()

        self.emb1 = nn.Embedding(w1_len, emb_size)
        self.emb2 = nn.Embedding(w2_len, emb_size)
        self.lstm = nn.LSTM(emb_size*2, lstm_hidden, batch_first=True)
        self.fc0  = nn.Linear(lstm_hidden*2, lstm_out_size)

        self.fc1 = nn.Linear(in_ch + lstm_out_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden*2)
        self.fc3 = nn.Linear(n_hidden*2, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 1)

        self.drop = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, word1, word2):
        wd1 = self.emb1(word1)
        wd2 = self.emb2(word2)
        wds = torch.cat([wd1, wd2], dim=1)
        _, hc = self.lstm(wds.view(len(wds), 1, -1))
        out = torch.cat([hc[0][0], hc[1][0]], dim=1)
        out = self.sigmoid(self.fc0(out))
        
        out = torch.cat([input, out], dim=1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.drop(out)
        out = self.sigmoid(self.fc4(out))

        return out