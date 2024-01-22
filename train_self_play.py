'''
Created on Aug 27, 2023

@author: iapalm
'''

from cnn_brain import CNNModel, CNNBrain
from metrics_brain import MetricsBrain
from local_manager import LocalManager
from utils import pad_board, state_to_tensor, get_device
from display_manager import console_display
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import os
from random_brain import RandomBrain

device = get_device()

# instantiate a game with a CNN brain versus a CNN brain (from the same checkpoint)
# play a game (using local manager)
# every move that causes the bot to win a game will be postitive, otherwise negative
    # save a copy of each board and prediction, and once a win or loss occurs apply bellman and retrain
# take an optimization step every n games
# validate against random, metric, cnn bots
# if performance is better, update opponent

torch.autograd.set_detect_anomaly(True)

class BoardsDataset(Dataset):
    def __init__(self, boards, scores):
        self.boards = boards
        self.scores = scores
        
    def __len__(self):
        return self.boards.shape[0]
    
    def __getitem__(self, index):
        return self.boards[index, :, :, :], self.scores[index, :]

class SelfPlay():
    def __init__(self, starting_brain, save_model_fn, load_model_fn, lr = 0.1, q_lr=0.1, discount_factor=0.95, start_episode=1, episodes=10, games_per_episode=10, epochs_per_episode=16, dim=25):
        self.brain = starting_brain
        self.save_model_fn = save_model_fn
        self.load_model_fn = load_model_fn
        self.lr = lr
        self.q_lr = q_lr
        self.discount_factor = discount_factor
        self.start_episode = start_episode
        self.episodes = episodes
        self.games_per_episode = games_per_episode
        self.epochs_per_episode = epochs_per_episode
        self.dim = dim
        self.best_win_pct = None
        
    def play(self):
        n_episodes = self.start_episode
        self.save_model_fn(self.brain.model, n_episodes)
        self.opponent_1 = self.load_model_fn(n_episodes, name="Terminator")
        self.metrics_opponent = MetricsBrain("Validator")
        
        all_players = (self.brain, self.opponent_1)
        
        while n_episodes < self.start_episode + self.episodes + 1:
            # start an episode
            print("Starting episode {} / {}".format(n_episodes, self.start_episode + self.episodes))
            boards = torch.zeros((0, 12, self.dim, self.dim)).to(device)
            scores = torch.zeros((0, 1)).to(device)
            
            n_games = 0
            while n_games < self.games_per_episode:
                player_boards = {p: torch.zeros((0, 12, self.dim, self.dim)).to(device) for p in (self.brain, self.opponent_1)}
                
                # start a game
                print("Starting game {} / {}".format(n_games + 1, self.games_per_episode))
                lm = LocalManager(6, 6, self.brain, other_bots=[self.opponent_1], display=None, max_turns=250)
                
                end_game = False
                winners = {}
                while not end_game and lm.turn_number < lm.max_turns:
                    end_game, winners, _ = lm.turn()
                    
                    for p in all_players:
                        state = state_to_tensor(lm.board.view_board(p, fog=True))
                        
                        player_boards[p] = torch.vstack((state, player_boards[p]))
                    #preds = torch.vstack((pred, preds)
                
                for p in all_players:
                    reward = 10 if (p in winners) else -10
                    print("result: {} reward in {} turns".format(reward, lm.turn_number))
                    
                    old_prediction = self.brain.model(player_boards[p], sigmoid=False).squeeze()
                    optimal_future = torch.ones((old_prediction.size(0))).to(device)
                    discount_factors = self.discount_factor**(old_prediction.size(0) - torch.arange(old_prediction.size(0))).to(device)
                    
                    # now we apply bellman's equation to the unactivated nn outputs and then apply activations
                    bellman_scores = F.sigmoid(old_prediction + self.q_lr * (reward + discount_factors * optimal_future - old_prediction))
                    
                    boards = torch.vstack((player_boards[p], boards))
                    scores = torch.vstack((bellman_scores.detach().unsqueeze(1), scores)) # n x 1 tensor
                
                n_games += 1
                
            # training step
            self.train(boards, scores)
            
            self.save_model_fn(self.brain.model, n_episodes)
            
            # validation step
            self.validate()
            
            n_episodes += 1
        
    def train(self, boards, scores):
        dataset = BoardsDataset(boards, scores)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(self.brain.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.brain.model.parameters(), lr=self.lr)
        self.brain.model.train()
        
        print("Starting training, {} training events".format(len(train_loader)))
        for epoch in range(self.epochs_per_episode):
            i = 0
            running_loss = 0.0
            
            for b, s in train_loader:
                b, s = b.to(device), s.to(device)
                optimizer.zero_grad()
                
                prediction = self.brain.model.forward(b)
                
                #print(prediction, s)
                loss = criterion(prediction, s)
                loss.backward()
                optimizer.step()
                i += b.size(0)
                
                running_loss += loss.item()
                
            print('Epoch: {}/{}.............'.format(epoch + 1, self.epochs_per_episode), end=' ')
            print("Loss: {:.4f}".format(running_loss / i))
            
    def validate(self, n_games=5):
        # play against CNNBrain first
        print("Playing against metrics brain...")
        
        cnn_wins = 0
        for i in range(n_games):
            lm = LocalManager(6, 6, primary_bot=self.brain, other_bots=[self.metrics_opponent], time_delay=0, max_turns=250)
            bot_win = (self.brain in lm.play())
            
            if bot_win:
                print("result: win in {} turns".format(lm.turn_number))
                cnn_wins += 1
            elif lm.turn_number == lm.max_turns:
                print("result: draw in {} turns".format(lm.turn_number))
            else:
                print("result: lose in {} turns".format(lm.turn_number))
        
        """
        # play against metricsbot
        metrics_opponent = MetricsBrain("Robocop")
        print("{} / {} wins".format(cnn_wins, n_games))
        print("Playing against MetricsBrain...")
        
        metrics_wins = 0
        for i in range(3):
            lm = LocalManager(6, 6, primary_bot=self.brain, other_bots=[metrics_opponent], time_delay=0, max_turns=100)
            bot_win = (self.brain in lm.play())
            
            if bot_win:
                print("result: win in {} turns".format(lm.turn_number))
                metrics_wins += 1
            elif lm.turn_number == lm.max_turns:
                print("result: draw in {} turns".format(lm.turn_number))
            else:
                print("result: lose in {} turns".format(lm.turn_number))
                
        random_opponent = RandomBrain("iRobot")
        print("{} / {} wins".format(metrics_wins, 3))
        print("Playing against RandomBrain...")
        
        random_wins = 0
        for i in range(3):
            lm = LocalManager(6, 6, primary_bot=self.brain, other_bots=[random_opponent], time_delay=0, max_turns=100)
            bot_win = (self.brain in lm.play())
            
            if bot_win:
                print("result: win")
                random_wins += 1
            elif lm.turn_number == lm.max_turns:
                print("result: draw")
            else:
                print("result: lose")
        
        print("{} / {} wins".format(random_wins, 3))
        """
        
        # TODO: fix criteria - needs to be more than just win pct as opponent will get better
        if self.best_win_pct is None or cnn_wins / n_games >= self.best_win_pct:
            print("Saving best model")
            self.best_win_pct = cnn_wins / n_games
            self.save_model_fn(self.brain.model, "best")
            
            self.opponent_1 = self.load_model_fn("best", "Terminator")
        

if __name__ == "__main__":
    experiment_name = "self-play-1"
    experiment_folder = "models/real_value_metrics/{}".format(experiment_name)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    
    def load_model(episode, name):
        return CNNBrain(from_checkpoint="models/real_value_metrics/{}/model_ep_{}.pt".format(experiment_name, episode), name=name)
    
    def save_model(model, episode):
        torch.save(model.state_dict(), "models/real_value_metrics/{}/model_ep_{}.pt".format(experiment_name, episode))
    
    train_start_episode = 0
    #bot = CNNBrain(from_checkpoint="models/cnnmodel_32e_50g_initial.pt", name="Robocop")
    bot = load_model("0", "Robocop")
    
    sp = SelfPlay(starting_brain=bot, save_model_fn=save_model, load_model_fn=load_model, lr=0.01, q_lr=0.1, discount_factor=0.997, start_episode=train_start_episode, episodes=20, games_per_episode=5, epochs_per_episode=32, dim=25)
    
    sp.play()