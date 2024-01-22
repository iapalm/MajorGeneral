'''
Created on Aug 25, 2023

@author: iapalm
'''

from metrics_brain import MetricsBrain
from random_brain import RandomBrain
from cnn_brain import CNNModel, CNNBrain
from metrics_nn import MetricsNN
from local_manager import LocalManager

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class BoardsDataset(Dataset):
    def __init__(self, boards, metrics, score):
        self.boards = boards
        self.metrics = metrics
        self.score = score
        
    def __len__(self):
        return self.boards.shape[0]
    
    def __getitem__(self, index):
        return self.boards[index, :, :, :], self.metrics[index, :], self.score[index, :]

def train_cnn_metrics(cnn_model, metrics_model, train_set, val_set, epochs=32, lr=0.01):
    criterion = nn.BCELoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    
    print("Starting training")
    lowest_val = None
    for epoch in range(epochs):
        i = 0
        running_loss = 0.0
        cnn_model.train()
        
        for b, m, s in train_set:
            b, m, s = b.float().to(device), m.float().to(device), s.to(device)
            optimizer.zero_grad()
            
            prediction = cnn_model.forward(b)
            
            prob = metrics_model.forward(m)
            confidence = 1 / (1 + torch.abs(s - prob))
            loss = criterion(prediction, prob)
            loss *= confidence
            
            #if i == 0:
                #print(b[1][6, :, :])
                #print("pred", prediction)
                #print("prob", prob)
                #print("conf", confidence)
                #print("loss", loss)
                #print("mean", torch.mean(loss))
            torch.mean(loss).backward()
            #raise ValueError()
            optimizer.step()
            i += b.size(0)
            
            running_loss += torch.mean(loss).item()
            
        print('Epoch: {}/{}.............'.format(epoch + 1, epochs), end=' ')
        print("Loss: {:.4f}".format(running_loss / i))
        
        cnn_model.eval()
        j = 0
        val_loss = 0
        for b, m, s in val_set:
            b, m, s = b.float().to(device), m.float().to(device), s.to(device)
            
            prediction = cnn_model.forward(b)
            prob = metrics_model.forward(m)
            confidence = 1 / (1 + torch.abs(s - prob))
            
            loss = criterion(prediction, prob)
            loss *= confidence
            j += b.size(0)
            
            val_loss += torch.mean(loss).item()
        
        print("Validation loss: {:.4f}".format(val_loss / j))
        
        if lowest_val is None or val_loss / j < lowest_val:
            lowest_val = val_loss / j
            torch.save(cnn_model.state_dict(), "models/real_value_metrics/cnnmodel_32e_200g_initial.pt")
            print("Model saved")
    
if __name__ == "__main__":
    rng = np.random.default_rng()
    
    lr = 0.001
    
    boards = torch.load("data/real_value_data/boards_200.pt")
    metrics = torch.load("data/real_value_data/metrics_200.pt")
    score = torch.load("data/real_value_data/score_200.pt")
    
    print("{} moves loaded".format(boards.shape[0]))
    
    cnn_model = CNNModel().float()
    cnn_model.to(device)
    
    #metrics_model = torch.load("models/metricsnn_32e_15g_model.pt")
    #torch.save(metrics_model.state_dict(), "models/metricsnn_32e_15g.pt")
    metrics_model = MetricsNN()
    metrics_model.load_state_dict(torch.load("models/real_value_metrics/metricsnn_32e_200g.pt"))
    metrics_model.to(device)
    metrics_model.eval()
    
    dataset = BoardsDataset(boards, metrics, score)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    
    print("Training samples: ", len(train_set))
    print("Validaiton samples: ", len(val_set))
    
    train_cnn_metrics(cnn_model, metrics_model, train_loader, val_loader, epochs=32, lr=lr)
    
    #torch.save(cnn_model.state_dict(), "models/cnnmodel_32e_50g_initial.pt")