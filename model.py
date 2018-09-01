import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, class_num, dropout, n_layers=1):
        super(BiLSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            #dropout=dropout,
            bidirectional=True
        )

        self.linear = nn.Linear(
            2 * hidden_size,
            2 * hidden_size
        )

        self.self_attn = LinearSeqAttn(
            2 * hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                class_num
            )
        )

    def forward(self, x):
        '''
        Inputs:
            x: [T * B]
        '''
        # [T * B] -> [T * B * E]
        embed = self.embedding(x)

        # [T * B * E] -> [T * B * 2H]
        output, hidden = self.lstm(embed)

        # [T * B * 2H] -> [B * T * 2H]
        output = output.transpose(0, 1).contiguous()

        # [B * T * 2H] -> [B * T * 2H]
        output = torch.tanh(self.linear(output)) 

        # weights [B * T]
        weights = self.self_attn(output)

        # [B * T] -> [B * 1 * T] * [B * T * 2H] -> [B * 1 * 2H] -> [B * 2H]
        output = weights.unsqueeze(1).bmm(output).squeeze(1)
        
        # [B * 2H] -> [B * num_class]
        output = self.classifier(output)

        return output



class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        """
        Args:
            x: [B * T * 2H]
        Output:
            alpha: [B * T]
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class BiGRUNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, class_num, dropout, n_layers=1):
        super(BiGRUNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            #dropout=dropout,
            bidirectional=True
        )

        self.linear1 = nn.Linear(
            hidden_size * 2,
            192
        )

        self.linear2 = nn.Linear(
            192,
            84
        )

        self.linear3 = nn.Linear(
            84,
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
        output, hidden = self.gru(embed, None)

        # [T * B * 2H] -> [B * T * 2H] 
        output = output.transpose(0, 1).contiguous()

        # sum [B * T * 2H] -> [B * 2H]
        output = torch.sum(output, dim=1)
        
        # [B * 2H] -> [B * 192] -> [B * 84] -> [B * 19]
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output

"""
class CNNNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, class_num, n_layers=1):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1
            ), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        ) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
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
        # [T * B] -> [T * B * E] -> [B * T * E] -> [B * 1 * T * E]
        x = self.embedding(x).transpose(0, 1).unsqueeze(1)

        # [B * 1 * 1024 * 64] -> [B * 16 * 512 * 32]
        x = self.conv1(x)  

        # [B * 16 * 512 * 32] -> [B * 32 * 256 * 16]
        x = self.conv2(x)  

        # [B * 32 * 256 * 16] -> [B * 64 * 128 * 8]
        x = self.conv3(x)  

        # [B * Tx2H] -> [B * 200]
        output = self.linear1(output)
        output = F.relu(output)

        # [B * 200] -> [B * class_num]
        output = self.linear2(output)

        return output
"""
