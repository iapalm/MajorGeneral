'''
Created on Aug 19, 2023

@author: iapalm
'''

from metrics_brain import MetricsBrain
from random_brain import RandomBrain
from metrics_nn import MetricsNN
from local_manager import LocalManager
from utils import pad_board

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

class MetricsDataset(Dataset):
    def __init__(self, metrics, score):
        self.metrics = metrics
        self.score = score
        
    def __len__(self):
        return self.metrics.shape[0]
    
    def __getitem__(self, index):
        return self.metrics[index, :], self.score[index, :]

def generate_dataset(n_games=100):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    boards = torch.zeros((0, 12, 25, 25)).to(device)
    metrics = torch.zeros((0, 15)).to(device)
    score = torch.zeros((0, 1)).to(device)
    
    for game in range(n_games):
        h = rng.integers(6, 10)
        w = rng.integers(6, 10)
        rb1 = MetricsBrain("robocop")
        rb2 = MetricsBrain("terminator")
        lm = LocalManager(h, w, primary_bot=rb1, other_bots=[rb2])
        
        end_game = False
        bot_win = False
        while not end_game and lm.turn_number < lm.max_turns:
            try:
                end_game, winners, m = lm.turn()
                boards = torch.vstack((boards, pad_board(torch.from_numpy(lm.board.view_board(lm.primary_bot)).to(device)).unsqueeze(0)))
                metrics = torch.vstack((metrics, torch.from_numpy(m[1:]).to(device)))
                score = torch.vstack((score, torch.tensor(sigmoid(m[0] / 100)).to(device)))
            except:
                print("Error")
                end_game = True
            
        #win = lm.play()
        
        print("Completed game {} of {}".format(game + 1, n_games))
        
    return boards, metrics, score


def train(model, train_set, val_set, epochs=32, lr=0.01):
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training")
    lowest_val = None
    for epoch in range(epochs):
        i = 0
        running_loss = 0.0
        model.train()
        for x, y in train_set:
            x, y = x.double().to(device), y.to(device)
            optimizer.zero_grad()
            
            prediction = model.forward(x)
            loss = criterion(prediction, y)
            loss.backward()
            
            optimizer.step()
            i += x.size(0)
            
            running_loss += loss.item()
            
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Loss: {:.4f}".format(running_loss / i))
        
        model.eval()
        j = 0
        val_loss = 0
        for x, y in val_set:
            x, y = x.double().to(device), y.to(device)
            
            prediction = model.forward(x)
            loss = criterion(prediction, y)
            j += x.size(0)
            
            val_loss += loss.item()
        
        print("Validation loss: {:.4f}".format(val_loss / j))
        
        if lowest_val is None or val_loss / j < lowest_val:
            lowest_val = val_loss / j
            torch.save(model.state_dict(), "models/real_value_metrics/metricsnn_32e_200g.pt")
            print("Model saved")

if __name__ == "__main__":
    rng = np.random.default_rng()
    load_data_from_file = True
    n_games = 200
    lr = 0.01
    
    if load_data_from_file:
        boards = torch.load("data/real_value_data/boards_200.pt")
        metrics = torch.load("data/real_value_data/metrics_200.pt")
        score = torch.load("data/real_value_data/score_200.pt")
    else:
        boards, metrics, score = generate_dataset(n_games)
        torch.save(boards, "data/real_value_data/boards_{}.pt".format(n_games))
        torch.save(metrics, "data/real_value_data/metrics_{}.pt".format(n_games))
        torch.save(score, "data/real_value_data/score_{}.pt".format(n_games))
    
    print("{} moves loaded".format(metrics.shape[0]))
    
    model = MetricsNN().double()
    model.to(device)
    
    dataset = MetricsDataset(metrics, score)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    
    train(model, train_set, val_set, epochs=32, lr=lr)