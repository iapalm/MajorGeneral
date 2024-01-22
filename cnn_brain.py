'''
Created on Jun 25, 2023

@author: iapalm
'''

from brain import Brain
from monte_carlo_search import MonteCarloSearch
from utils import pad_board, state_to_tensor
from node import Node

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

## map state format
# layer 0: empty tiles (1 where empty, 0 elsewhere)
# layer 1: mountain tiles (1 where mountain, 0 elsewhere)
# layer 2: fog tiles (1 where fog, 0 elsewhere)
# layer 3: fog obstacle tiles (1 where fog obstacle, 0 elsewhere)
# layer 4: cities (1 where city exists, 0 elsewhere)
# layer 5: generals (1 where general exists, 0 elsewhere)
# layer 6: bot armies (number representing quantity)
# layer 7: neutral armies (cities, number representing quantity)
# layer 8-10: other player armies (number representing quantity)
# layer 11: all other armies (number representing quantity)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(12, 14, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(14, 8, 5)
        self.fc1 = nn.Linear(72, 24)
        self.fc2 = nn.Linear(24, 1)
    
    def forward(self, x, sigmoid=True):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        
        if sigmoid:
            return F.sigmoid(self.fc2(out))
        else:
            return self.fc2(out)
        

class CNNBrain(Brain):
    def __init__(self, name, from_checkpoint=None):
        super(CNNBrain, self).__init__(name)
        
        if from_checkpoint is not None:
            self.model = CNNModel().to(device)
            
            self.model.load_state_dict(torch.load(from_checkpoint))
            
    def copy(self):
        b = Brain.copy(self)
        b.model = self.model # reference only - don't clone entire model
        return b
            
    
    def turn(self, state, metrics, fog_board):
        max_turn_time = 0.45 # seconds
        
        start_turn_time = time.time()
        
        def sample_turn(board, move):
            new_board = board.copy()
            new_board.process_turn(((self, move),), new_board.turn + 1)
            return new_board
        
        def valid_move(state, end_i, end_j):
            if end_i < 0 or end_i >= state.shape[1]:
                return False
            elif end_j < 0 or end_j >= state.shape[2]:
                return False
            elif np.sum(state[1:4, end_i, end_j], axis=0) > 0:
                return False
            else:
                return True
        
        rng = np.random.default_rng()
        
        _, map_h, map_w = state.shape
        
        def eval_state(state):
            state_tensor = state_to_tensor(state)
            return self.model(state_tensor, sigmoid=True).item()
        
        potential_moves = [Node(None, fog_board, eval_state(state))] # list of nodes, sorted with high win % in front
        
        while time.time() - start_turn_time < max_turn_time:
            # select node to explore
            i = 0
            node_to_explore = None
            while i < len(potential_moves):
                if not potential_moves[i].explored:
                    node_to_explore = potential_moves[0]
                    break
                i += 1
                
            # if all nodes are explored, pick best move
            if node_to_explore is None:
                break
            
            # otherwise, explore node
            node_to_explore.explored = True
            
            node_board = node_to_explore.resulting_board
            node_state = node_board.view_board(self)
            
            potential_start_tiles = np.vstack(np.where(node_state[6] >= 2))
        
            # if no possible moves, continue
            if potential_start_tiles.shape[1] == 0:
                continue
            
            # expand node
            for i in range(potential_start_tiles.shape[1]):
                start_tile = potential_start_tiles[:, i]
                start_index = start_tile[0] * map_w + start_tile[1]
                target_indices = []
            
                if valid_move(node_state, start_tile[0], start_tile[1] - 1): # left
                    target_indices.append(start_index - 1)
                if valid_move(node_state, start_tile[0], start_tile[1] + 1): # right
                    target_indices.append(start_index + 1)
                if valid_move(node_state, start_tile[0] + 1, start_tile[1]): # down
                    target_indices.append(start_index + map_w)
                if valid_move(node_state, start_tile[0] - 1, start_tile[1]): # up
                    target_indices.append(start_index - map_w)
                
                for target_index in target_indices:
                    is50_options = (True, False) if node_state[6, start_tile[0], start_tile[1]] > 2 else (False,)
                    for is50 in is50_options:
                        move = {"start": int(start_index), "end": int(target_index), "is50": is50}
                        resulting_board = sample_turn(node_board, move)
                        potential_move = Node(move, resulting_board, eval_state(resulting_board.view_board(self)))
                        
                        # insert into the list
                        i = 0
                        while i < len(potential_moves):
                            if potential_move.p_win > potential_moves[i].p_win:
                                potential_moves.insert(i, potential_move)
                                break
                            i += 1
                        if i == len(potential_moves):
                            potential_moves.append(potential_move)
        
        #print("Explored {} total nodes".format(len(potential_moves)))
        return potential_moves[0].move
    
class CNNMonteCarloBrain(CNNBrain):
    def __init__(self, name, from_checkpoint=None):
        super(CNNMonteCarloBrain, self).__init__(name, from_checkpoint)
        self.search = MonteCarloSearch()
        
        def eval_board(board):
            state = board.view_board(self)
            state_tensor = state_to_tensor(state)
            return self.model(state_tensor, sigmoid=True).item()
        
        self.eval_fn = eval_board
        
    def turn(self, state, metrics, fog_board):
        max_turn_time = 0.45 # seconds
        
        return self.search.turn(state, fog_board, self, self.eval_fn, max_turn_time, reset_root=fog_board.turn % 10 == 0)