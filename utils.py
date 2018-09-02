import pandas as pd
import torch
import json
import random
import pickle

class Daguan(object):
    def __init__(self, config):
        self.config = config
        self.word_to_id = None

    def load_dataset(self):
        df = pd.read_csv("new_data/train_set.csv")

        dataset = []
        labels = []
        
        for _, row in df.iterrows():
            words = row['word_seg'].split()

            dataset.append(words)
            labels.append(int(row['class']) - 1)

        assert(len(dataset) == len(labels))

        # shuffle
        c = list(zip(dataset, labels))
        random.shuffle(c)
        dataset, labels = zip(*c)

        with open('word_to_id.pkl', 'rb') as f:
            self.word_to_id = pickle.load(f)

        self.config.model.vocab_size = len(self.word_to_id)
        return dataset, labels, self.word_to_id
    
    def load_test_dataset(self):
        df = pd.read_csv("new_data/test_set.csv")

        dataset = []

        try:
            with open("word_to_id.json", "r") as f:
                self.word_to_id = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("word_to_id.json is not found")

        for _, row in df.iterrows():
            words = row['word_seg'].split()
            dataset.append(words)

        assert(self.config.model.vocab_size == len(self.word_to_id))
        return dataset, self.word_to_id


def sent_to_tensor(batch, word_to_id, max_seq_len):
    '''
    Inputs:
        batch: [B * T]   type:string list
    Outpus:
        tensor: [T * B]  type:tensor
    '''
    batch_size = len(batch)
    
    tensor = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
    for i in range(batch_size):
        min_len = min(len(batch[i]), max_seq_len)
        for j in range(min_len):
            id = word_to_id.get(batch[i][j], 0)
            tensor[j][i] = id

    return tensor
