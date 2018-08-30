import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

from config import Config
from model import *
from utils import Daguan, sent_to_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_CACHE_PATH"] = "/home/zyc/cudacache"

def train(config):
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    daguan = Daguan(config)
    dataset, labels, word_to_id = daguan.load_dataset()
    config.model.class_num = len(set(labels))
    print('class num:',config.model.class_num)

    size = len(dataset)
    print('data size:', size)
    divid = int(0.9 * size)
    train_dataset = dataset[:divid]
    train_labels = labels[:divid]
    dev_dataset = dataset[divid:]
    dev_labels = labels[divid:]

    print("Training from scratch!")
    net = BiLSTMNet(config.model.vocab_size,
            config.model.embedd_size,
            config.model.hidden_size,
            config.model.max_seq_len,
            config.model.class_num,
            config.model.n_layers)
    net.to(device)
    
    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.training.lr, 
                weight_decay=config.training.weight_decay) # L2
    
    
    print("Start training!")
    best_f1 = 0.0
    for epoch in range(config.training.epochs):
        total_loss = 0.0
        start = time.time()

        result = []
        # train
        for i in tqdm(range(0, len(train_dataset), batch_size)):
            optimizer.zero_grad()

            x = train_dataset[i: i + batch_size]
            label = train_labels[i: i + batch_size]

            x = sent_to_tensor(x, word_to_id,  max_seq_len).to(device)
            label = torch.LongTensor(label).to(device)
            
            output = net(x)
            result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        f1, precision, recall, _  = score(train_labels, result, average='macro')
        print("Epoch %d: train f1 score: %.2f  precision: %.2f  recall: %.2f" % (epoch, 100 * f1, 
            100 * precision, 100 * recall))
        print("Epoch %d train loss: %.3f  time: %.3f s" % (epoch, total_loss / len(train_dataset), time.time() - start))

        # dev
        with torch.no_grad():
            result = []
            for i in range(0, len(dev_dataset), batch_size):
                x = dev_dataset[i: i + batch_size]
                label = dev_labels[i: i + batch_size]
                
                x = sent_to_tensor(x, word_to_id, max_seq_len).to(device)
                label = torch.LongTensor(label).to(device)

                output = net(x)
                result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

        # F1 score
        f1, precision, recall, _  = score(dev_labels, result, average='macro')
        print("Epoch %d: dev f1 score: %.2f  precision: %.2f  recall: %.2f" % (epoch, 100 * f1, 
            100 * precision, 100 * recall))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(net, config.resourses.model_path + "_" + config.resourses.model_name)
            print("net saved!")


    
if __name__=="__main__":
    config = Config()
    train(config)
