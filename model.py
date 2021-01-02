import torch
from torch import nn

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data)[0][idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]

        return out_data, out_label


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
