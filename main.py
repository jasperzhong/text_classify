import argparse
import os
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

from config import Config
from model import *
from utils import Daguan, sent_to_tensor

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

    # load pretrained embedding
    with open('word2vec.pkl', 'rb') as f:
        embedding = torch.FloatTensor(pickle.load(f))
    
    print("Training from scratch!")
    if config.model.module == "BiLSTM":
        net = BiLSTMNet(config.model.vocab_size,
                config.model.embedd_size,
                config.model.hidden_size,
                config.model.max_seq_len,
                config.model.class_num,
                config.model.dropout,
                embedding,
                config.training.fix,
                config.model.n_layers)
    elif config.model.module == "BiGRU":
        net = BiGRUNet(config.model.vocab_size,
                config.model.embedd_size,
                config.model.hidden_size,
                config.model.max_seq_len,
                config.model.class_num,
                config.model.dropout,
                config.model.n_layers)
    elif config.model.module == "BiLSTMSum":
        net = BiLSTMSumNet(config.model.vocab_size,
                config.model.embedd_size,
                config.model.hidden_size,
                config.model.max_seq_len,
                config.model.class_num,
                config.model.dropout,
                embedding,
                config.training.fix,
                config.model.n_layers)
    else:
        raise ValueError("Undefined network")

    net.to(device)
    
    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), 
                lr=config.training.lr, 
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
        
        precision, recall, f1, _  = score(train_labels, result, average='macro')
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
        precision, recall, f1, _  = score(dev_labels, result, average='macro')
        print("Epoch %d: dev f1 score: %.2f  precision: %.2f  recall: %.2f" % (epoch, 100 * f1, 
            100 * precision, 100 * recall))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(net, config.resourses.model_path + "_" + config.resourses.model_name)
            print("net saved!")


def test(config):
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    daguan = Daguan(config)
    dataset, word_to_id = daguan.load_dataset()
    
    try:
        net = torch.load(config.resourses.model_path + "_" + config.resourses.model_name)
    except FileNotFoundError:
        raise FileNotFoundError("No model!")
    
    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len
    # test
    with torch.no_grad():
        result = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            x = dataset[i: i + batch_size]
            x = sent_to_tensor(x, word_to_id, max_seq_len).to(device)

            output = net(x)
            result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 
    result = [n + 1 for n in result]
    id = [i for i in range(len(result))]
    df = pd.DataFrame({"id":id, "class": result})
    df.to_csv("result.csv", index=False, encoding='utf-8')


if __name__=="__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-module', type=str, choices=config.model.MODULES)
    parser.add_argument('-gpu', type=str, default=0)
    parser.add_argument('-model_name', type=str, default=config.resourses.model_name)
    parser.add_argument('-fix', type=bool, default=True)
    args = parser.parse_args()

    config.model.module = args.module
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config.resourses.model_name = args.model_name
    config.training.fix = args.fix
    config.mode = args.mode

    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
    else:
        raise ValueError("undefined mode")
