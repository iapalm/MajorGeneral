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
# layer 8: friendly armies (number representing quantity)
# layer 9-10: other player armies (number representing quantity)
# layer 11: all other armies (number representing quantity)

class RandomBrain(Brain):
    def turn(self, state, sample_turn_fn):
        rng = np.random.default_rng()
        
        _, map_h, map_w = state.shape
        
        potential_moves = np.vstack(np.where(state[6] >= 2))
        
        if potential_moves.shape[1] == 0:
            #print("No potential moves")
            return None
        
        start_tile = rng.choice(potential_moves, axis=1)
        #print(start_tile)
        start_index = start_tile[0] * map_w + start_tile[1]
        #print(start_index)
        #print(state[2, start_tile[0], start_tile[1]])
        #print(state[6, start_tile[0], start_tile[1]])
        target_indices = []
        
        if start_tile[1] > 0 and np.sum(state[1:4, start_tile[0], start_tile[1] - 1], axis=0) == 0: # left
            target_indices.append(start_index - 1)
        if start_tile[1] < map_w - 1 and np.sum(state[1:4, start_tile[0], start_tile[1] + 1], axis=0) == 0: # right
            target_indices.append(start_index + 1)
        if start_tile[0] < map_h - 1 and np.sum(state[1:4, start_tile[0] + 1, start_tile[1]], axis=0) == 0: # down
            target_indices.append(start_index + map_w)
        if start_tile[0] > 0 and np.sum(state[1:4, start_tile[0] - 1, start_tile[1]], axis=0) == 0: # up
            target_indices.append(start_index - map_w)
        
        if len(target_indices) == 0:
            #print("No valid moves")
            return None
        else:
            target_index = rng.choice(target_indices)
            
            return {"start": int(start_index), "end": int(target_index), "is50": False}