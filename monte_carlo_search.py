'''
Created on Sep 16, 2023

@author: iapalm
'''

from node import Node

import numpy as np
import time

class MonteCarloSearch():
    def __init__(self):
        self.time_scale_factor = 1 # even balance of exploit/explore
        self.root_node = None
        
    def turn(self, state, fog_board, player, eval_fn, max_turn_time=0.45, reset_root=False):
        start_turn_time = time.time()
        
        
        def sample_turn(board, move):
            new_board = board.copy()
            new_board.process_turn(((player, move),), new_board.turn + 1)
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
        
        _, map_h, map_w = state.shape
        
        if reset_root or self.root_node is None:
            root_node = Node(None, fog_board, eval_fn(fog_board))
        else:
            root_node = self.root_node
        
        iters = 0
        while time.time() - start_turn_time < max_turn_time:
            # select node to explore
            iters += 1
            i = 0
            node_to_explore = root_node.select_to_explore(iters)
                
            # if all nodes are explored, pick best move
            if node_to_explore is None:
                print("done exploring")
                break
            
            # otherwise, explore node
            node_to_explore.visit()
            
            node_board = node_to_explore.resulting_board
            node_state = node_board.view_board(player)
            
            potential_start_tiles = np.vstack(np.where(node_state[6] >= 2))
            
            # no move is an option
            #if node_to_explore.move is not None:
            move = None
            resulting_board = sample_turn(node_board, move)
            _ = Node(move, resulting_board, eval_fn(resulting_board), parent=node_to_explore, c=self.time_scale_factor)
            
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
                        _ = Node(move, resulting_board, eval_fn(resulting_board), parent=node_to_explore, c=self.time_scale_factor)
                        #_ = Node(move, resulting_board, eval_fn(resulting_board), parent=node_to_explore, c=0)
                        
        if root_node.height() < 5 or root_node.get_best_child().move is None: # not looking 5 turns ahead
            self.time_scale_factor /= 1.5 # exploit more
#             print("Shrinking to {}".format(self.time_scale_factor))
        elif root_node.height() > 10: # looking too far ahead
            self.time_scale_factor *= 1.5 # explore more
#             print("Increase to {}".format(self.time_scale_factor))
        
        #print("best metric {}".format(root_node.get_best_child()))
#         if fog_board.turn > 40:
#             print(iters)
#             print(root_node.__len__())
#             print(root_node.height())
#             print(root_node.get_best_child().move)
#             print(root_node.print_tree(3))
#             raise ValueError()
        #print("Explored {} total nodes".format(len(potential_moves)))
        self.root_node = root_node.get_best_child()
        self.root_node.parent = None
        return root_node.get_best_child().move