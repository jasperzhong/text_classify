import pandas as pd
import torch
from tqdm import tqdm


class Dictionary(object):
    def __init__(self):
        self.word_to_cnt = {}
    
    def __len__(self):
        return len(self.word_to_cnt)

    def add(self, word_ix):
        if not self.word_to_cnt.get(word_ix):
            self.word_to_cnt[word_ix] = 0
        self.word_to_cnt[word_ix] += 1


def load_dataset():
    df = pd.read_csv("new_data/train_set.csv")

    dictionary = Dictionary()
    dataset = []
    labels = []

    for _, row in tqdm(df.iterrows()):
        words = row['word_seg'].split()
        words = [int(word) for word in words]

        dataset.append(words)
        labels.append(int(row['class']))
        for word in words:
            dictionary.add(word)

    print('Vocab size is: ', len(dictionary))

    return dataset, labels, len(dictionary)


def sent_to_tensor(batch, max_seq_len):
    '''
    Inputs:
        batch: [B * T]   type:index list
    Outpus:
        tensor: [T * B]  type:tensor
    '''
    batch_size = len(batch)
    
    tensor = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
    for i in range(batch_size):
        min_len = min(len(batch[i]), max_seq_len)
        for j in range(min_len):
            tensor[j][i] = batch[i][j]

    return tensor
