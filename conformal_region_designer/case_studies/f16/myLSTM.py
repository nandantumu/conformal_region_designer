import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMNew(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        lstm_hidden_size,
        hidden_size,
        lstm_num_layers,
        device,
    ):
        super(LSTMNew, self).__init__()

        self.num_classes = num_classes
        self.num_layers = lstm_num_layers
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
        #                     num_layers=num_layers, batch_first=True)

        self.linear1 = nn.Linear(lstm_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward_stupid(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = Variable(
                torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size)
            ).to(self.device)
        if c_0 is None:
            c_0 = Variable(
                torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size)
            ).to(self.device)

        # Propagate input through LSTM
        ula, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        ula = ula.view(x.size(0), x.size(1), -1)
        ula = F.relu(self.linear1(ula))
        ula = F.relu(self.linear2(ula))
        out = self.fc(ula)

        return out, h_out, c_out
