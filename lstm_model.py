import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, batch_size, num_layers=1):
        super(SentimentLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, input, hidden, cell):
        out,(hidden, cell) = self.lstm(input,(hidden, cell))
        out_x = out.clone()
        out_x = self.fc(out_x)
        sig_out = self.sig(out_x)

        # reshape to be batch_size first
        sig_out = sig_out.view(self.batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden, cell

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        cell = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, cell

