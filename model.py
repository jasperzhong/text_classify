import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, class_num, n_layers=1):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            n_layers, 
            bidirectional=True
        )

        self.linear1 = nn.Linear(
            hidden_size * 2 * seq_len,
            3000
        )

        self.linear2 = nn.Linear(
            3000,
            class_num
        )

    def forward(self, x):
        '''
        Inputs:
            x: [T * B]
        '''
        # [T * B] -> [T * B * E]
        embed = self.embedding(x)

        # [T * B * E] -> [T * B * 2H]
        output, hidden = self.lstm(embed, None)

        # [T * B * 2H] -> [B * T * 2H] -> [B * Tx2H]
        output = output.transpose(0, 1).contiguous()
        output = output.view(output.size(0), -1)

        # [B * Tx2H] -> [B * 3000]
        output = self.linear1(output)
        output = F.relu(output)

        # [B * 3000] -> [B * class_num]
        output = self.linear2(output)

        return output

"""
class CNNNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, class_num, n_layers=1):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.conv1 = nn.Conv2d(

        )

        self.linear1 = nn.Linear(
            hidden_size * 2 * seq_len,
            200
        )

        self.linear2 = nn.Linear(
            200,
            class_num
        )

    def forward(self, x):
        '''
        Inputs:
            x: [T * B]
        '''
        # [T * B] -> [T * B * E]
        embed = self.embedding(x)

        # [T * B * E] -> [T * B * 2H]
        output, hidden = self.lstm(embed, None)

        # [T * B * 2H] -> [B * T * 2H] -> [B * Tx2H]
        output = output.transpose(0, 1).contiguous()
        output = output.view(output.size(0), -1)

        # [B * Tx2H] -> [B * 200]
        output = self.linear1(output)
        output = F.relu(output)

        # [B * 200] -> [B * class_num]
        output = self.linear2(output)

        return output
"""