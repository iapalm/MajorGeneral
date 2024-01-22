'''
Created on Jun 25, 2023

@author: iapalm
'''

from brain import Brain
from monte_carlo_search import MonteCarloSearch
from node import Node
import numpy as np
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

class MetricsBrain(Brain):
    def turn(self, state, metrics, fog_board):
        def sample_turn(board, move):
            new_board = board.copy()
            new_board.process_turn(((self, move),), new_board.turn + 1)
            return (new_board.view_board(self, fog=True), new_board.get_metrics(self))
    
        rng = np.random.default_rng()
        
        _, map_h, map_w = state.shape
        
        potential_start_tiles = np.vstack(np.where(state[6] >= 2))
        
        if potential_start_tiles.shape[1] == 0:
            return None
        
        potential_move_metrics = [(None, metrics)] # (move, (metrics tuple))
        
        for i in range(potential_start_tiles.shape[1]):
            start_tile = potential_start_tiles[:, i]
            start_index = start_tile[0] * map_w + start_tile[1]
            target_indices = []
        
            if start_tile[1] > 0 and np.sum(state[1:4, start_tile[0], start_tile[1] - 1], axis=0) == 0: # left
                target_indices.append(start_index - 1)
            if start_tile[1] < map_w - 1 and np.sum(state[1:4, start_tile[0], start_tile[1] + 1], axis=0) == 0: # right
                target_indices.append(start_index + 1)
            if start_tile[0] < map_h - 1 and np.sum(state[1:4, start_tile[0] + 1, start_tile[1]], axis=0) == 0: # down
                target_indices.append(start_index + map_w)
            if start_tile[0] > 0 and np.sum(state[1:4, start_tile[0] - 1, start_tile[1]], axis=0) == 0: # up
                target_indices.append(start_index - map_w)
            
            for target_index in target_indices:
                for is50 in (True, False):
                    move = {"start": int(start_index), "end": int(target_index), "is50": is50}
                    potential_move_metrics.append((move, sample_turn(fog_board, move)[1]))
        
        rng.shuffle(potential_move_metrics)
        
        best_move, best_metric = max(potential_move_metrics, key=lambda m: m[1][0]) # select score out of metrics
        #print("Best metric ", eval_metrics(best_metric))
        return best_move
    
class MetricsMonteCarloBrain(Brain):
    def __init__(self, name):
        super(MetricsMonteCarloBrain, self).__init__(name)
        self.search = MonteCarloSearch()
        
        def eval_board(board):
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            return sigmoid(board.get_metrics(self)[0] / 50) # score
        
        self.eval_fn = eval_board
        
    def turn(self, state, metrics, fog_board):
        max_turn_time = 0.45 # seconds
        
        return self.search.turn(state, fog_board, self, self.eval_fn, max_turn_time, reset_root=fog_board.turn % 10 == 0)