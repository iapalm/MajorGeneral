'''
Created on Jun 25, 2023

@author: iapalm
'''

from brain import Brain
import numpy as np

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
    def turn(self, state, sample_turn_fn):
        rng = np.random.default_rng()
        
        _, map_h, map_w = state.shape
        
        potential_start_tiles = np.vstack(np.where(state[6] >= 2))
        
        if potential_start_tiles.shape[1] == 0:
            return None
        
        def get_metrics(state):
            # territory, armies, enemies, neutrals, 1 where control cities, number of units on general
            return (np.sum(state[6] > 0), np.sum(state[6]), np.sum(state[8:]), np.sum(state[7]), np.sum((state[4] * state[6]) > 0), np.sum(state[5] * state[6]))
        
        def eval_metrics(metrics):
            return metrics[0] + metrics[1] - 0.8 * metrics[2] - 0.5 * metrics[3] + 50 * metrics[4] #+ 0.1 * metrics[5] # total + territory - 0.8 * enemies - 0.5 * neutrals + 50 * cities + 0.1 * unts on general
        
        potential_move_metrics = [(None, get_metrics(state))] # (move, (metrics tuple))
        
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
                move = {"start": int(start_index), "end": int(target_index), "is50": False}
                potential_move_metrics.append((move, get_metrics(sample_turn_fn(move))))
        
        rng.shuffle(potential_move_metrics)
        
        best_move, best_metric = max(potential_move_metrics, key=lambda m: eval_metrics(m[1]))
        #print("Best metric ", eval_metrics(best_metric))
        return best_move