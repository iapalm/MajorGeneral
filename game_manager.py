'''
Created on Jun 24, 2023

@author: iapalm
'''

import numpy as np
from random_brain import RandomBrain
from metrics_brain import MetricsBrain
from board import Board


class GameManager():
    TILE_EMPTY = -1
    TILE_MOUNTAIN = -2
    TILE_FOG = -3
    TILE_FOG_OBSTACLE = -4
    
    def __init__(self):
        #self.player_index = player_index
        #self.brain = RandomBrain("robocop")
        self.brain = MetricsBrain("robocop")
        self.generals = 0
        self.cities = []
        self.map = []
    
    def set_player_index(self, player_index):
        self.player_index = player_index
    
    @staticmethod
    def patch(old, diff):
        out = []
        i = 0
        
        while i < len(diff):
            if not diff[i] == 0: # matching
                out += old[len(out): len(out) + diff[i]]
            i += 1
            if i < len(diff) and not diff[i] == 0: # mismatching 
                out += diff[i + 1: i + 1 + diff[i]]
                i += diff[i]
            i += 1
        return out
    
    def sample_turn(self, player, board, move):
        new_board = board.copy()
        new_board.move(player, move)
        return new_board.view_board(player, fog=True)
    
    def process_turn(self, map_state):
        board = Board.from_game_state(map_state)
        return self.brain.turn(map_state, lambda move: self.sample_turn(self.brain, board, move))
    
    def update(self, data):
        self.map = self.patch(self.map, data["map_diff"])
        self.cities = self.patch(self.cities, data["cities_diff"])
        self.generals = data["generals"]
        
        map_w = self.map[0]
        map_h = self.map[1]
        size = map_w * map_h
        
        armies = self.map[2: size + 2]
        terrain = self.map[size + 2: size + 2 + size]
        
        terrain_mat = np.array(terrain).reshape((map_h, map_w))
        
        #player_filter = (terrain_mat >= 0) * ((terrain_mat == self.player_index) * 2 - 1)
        #armies_mat = np.array(armies).reshape((map_h, map_w)) * player_filter
        
        ## map state format
        # layer 0: empty tiles (1 where empty, 0 elsewhere)
        # layer 1: mountain tiles (1 where mountain, 0 elsewhere)
        # layer 2: fog tiles (1 where fog, 0 elsewhere)
        # layer 3: fog obstacle tiles (1 where fog obstacle, 0 elsewhere)
        # layer 4: cities (1 where city exists, 0 elsewhere)
        # layer 5: generals (1 where general exists, 0 elsewhere)
        # layer 6: bot armies (number representing quantity)
        # layer 7-10: other player armies (number representing quantity)
        # layer 11: all other armies (number representing quantity)
        
        map_state = np.zeros((12, map_h, map_w))
        map_state[0] = (terrain_mat == -1) * 1
        map_state[1] = (terrain_mat == -2) * 1
        map_state[2] = (terrain_mat == -3) * 1
        map_state[3] = (terrain_mat == -4) * 1
        
        cities_mat = np.zeros((map_h * map_w, 1))
        if len(self.cities) > 0:
            cities = np.array(self.cities)
            cities_mat[cities] = 1
        map_state[4] = cities_mat.reshape(map_h, map_w) # ones where cities exist, 0 otherwise
        
        
        generals = np.array([x for x in self.generals if x >= 0])
        generals_mat = np.zeros((map_h * map_w, 1))
        generals_mat[generals] = 1
        map_state[5] = generals_mat.reshape(map_h, map_w)
        
        # bot armies
        map_state[6] = (terrain_mat >= 0) * (terrain_mat == self.player_index) * np.array(armies).reshape((map_h, map_w))
        
        # next 4 players
        players_added = 0
        other_index = 0
        while players_added < 4:
            if not other_index == self.player_index:
                map_state[players_added + 7] = (terrain_mat >= 0) * (terrain_mat == other_index) * np.array(armies).reshape((map_h, map_w))
                players_added += 1
            other_index += 1
        
        map_state[11] = (terrain_mat >= 0) * (terrain_mat >= other_index) * (terrain_mat != self.player_index) * np.array(armies).reshape((map_h, map_w))
        
        #map_state = np.concatenate((np.expand_dims(terrain_mat, 0), np.expand_dims(armies_mat, 0)))
        
        #game_state = {"map_state": map_state, "t"}
        
        return map_state
        
    
#gm = GameManager()

#print(gm.patch([0, 0], [1, 1, 3]))
#print(gm.patch([0, 0], [0, 1, 2, 1]))
#print(gm.patch([], [0]))
#print(gm.patch([0, 686, 19, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -3, -3, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -4, -3, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -4, -3, -3, -4, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -4, -3, -3, -4, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -4, -3, -4, -4, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -4, -3, -3, -3, -4, -4, -4, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -1, -1, -2, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -4, -3, -4, -3, -2, 1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -4, -3, -3, -4, -3, -2, -1, -1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -4, -4], [308, 1, 2, 377]))