import pandas as pd
import torch
import json


class Daguan(object):
    def __init__(self, config):
        self.config = config
        self.word_to_id = {"<pad>": 0, "</s>": 1}
        self.word_cnt = {}
        self.vocab_size = 2

    def load_dataset(self):
        df = pd.read_csv("new_data/train_set.csv")

        dataset = []
        labels = []
        
        flag = 0
        try:
            with open("word_to_id.json", "r") as f:
                self.word_to_id = json.load(f)
        except FileNotFoundError:
            flag = 1

        for _, row in df.iterrows():
            words = row['word_seg'].split()

            dataset.append(words)
            labels.append(int(row['class']) - 1)
            if flag:
                for word in words:
                    if not self.word_cnt.get(word):
                        self.word_cnt[word] = 0
                    self.word_cnt[word] += 1

        assert(len(dataset) == len(labels))

        # shuffle
        c = list(zip(dataset, labels))
        random.shuffle(c)
        dataset, labels = zip(*c)

        if flag:
            self.word_cnt = sorted(self.word_cnt.items(), key = lambda x:int(x[1]), reverse=True)
            self.word_cnt = self.word_cnt[:self.config.model.top_words]

            for key, _ in self.word_cnt:
                self.word_to_id[key] = self.vocab_size
                self.vocab_size += 1

            with open("word_to_id.json", "w") as f:
                json_info = json.dumps(self.word_to_id)
                f.write(json_info)

        assert(self.config.model.vocab_size == len(self.word_to_id))
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
