'''
Created on Jun 24, 2023

@author: iapalm
'''

import numpy as np
from random_brain import RandomBrain
from metrics_brain import MetricsBrain
from human_brain import HumanBrain
from board import Board

import os


class GameManager():
    TILE_EMPTY = -1
    TILE_MOUNTAIN = -2
    TILE_FOG = -3
    TILE_FOG_OBSTACLE = -4
    
    def __init__(self, display=None):
        #self.player_index = player_index
        #self.brain = RandomBrain("robocop")
        self.brain = MetricsBrain("robocop")
        self.map = []
        self.generals = []
        self.cities = []
        self.players = [self.brain]
        self.board = None
        self.display = display
    
    def set_player_index(self, player_index):
        self.brain.set_index(player_index)
        
    def add_players(self, usernames, teams):
        players = []
        for i in range(len(usernames)):
            if i == self.brain.get_index():
                if len(teams) > 0:
                    self.brain.set_team(teams[i])
                else:
                    self.brain.set_team(i)
                players.append(self.brain)
            else:
                brain = HumanBrain(usernames[i])
                brain.set_index(i)
                if len(teams) > 0:
                    brain.set_team(teams[i])
                else:
                    brain.set_team(i)
                players.append(brain)
        
        self.players = players
    
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
    
    def sample_turn(self, player, move):
        new_board = self.board.copy()
        new_board.move(player, move)
        return new_board.view_board(player, fog=True)
    
    def process_turn(self, data):
        self.update(data)
        if self.display is not None:
            os.system("cls")
            self.display(self.board.view_board(self.brain))
        
        return self.brain.turn(self.board.view_board(self.brain, fog=True), lambda move: self.sample_turn(self.brain, move))
    
    def update(self, data):
        self.map = self.patch(self.map, data["map_diff"])
        self.cities = self.patch(self.cities, data["cities_diff"])
        self.generals = data["generals"]
        
        #print(self.map)
        #print(self.cities)
        
        map_w = self.map[0]
        map_h = self.map[1]
        
        if self.board is None:
            self.board = Board(map_h, map_w, players=self.players, generate=False)
        
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
        # layer 7: neutral armies (cities, number representing quantity)
        # layer 8: friendly armies (number representing quantity)
        # layer 9-10: other player armies (number representing quantity)
        # layer 11: all other armies (number representing quantity)
        
        cities_mat = np.zeros((map_h, map_w))
        for c in self.cities:
            i = c // map_w
            j = c % map_w
            cities_mat[i, j] = 1
        self.board.static_game_layers[Board.INDEX_CITIES] = cities_mat # ones where cities exist, 0 otherwise
        self.board.static_game_layers[Board.INDEX_NEUTRAL_ARMIES] = (terrain_mat == -1) * (cities_mat) * np.array(armies).reshape((map_h, map_w))
        
        self.board.static_game_layers[Board.INDEX_EMPTY] = (terrain_mat == Board.TILE_EMPTY) * (1 - cities_mat) * 1
        self.board.static_game_layers[Board.INDEX_MOUNTAIN] = (terrain_mat == Board.TILE_MOUNTAIN) * 1
        
        # the only fog layer we care about is the brain's because we assume that no other bot will require a view of the board
        self.board.player_fog_layers[self.brain] = (terrain_mat == Board.TILE_FOG) * 1
        
        # we make a big simplifying assumption that all fog obstacles are mountains here. it makes calculating the fog obstacle layer way easier. trust me. 
        self.board.static_game_layers[Board.INDEX_MOUNTAIN] += (terrain_mat == Board.TILE_FOG_OBSTACLE) * 1
        
        generals_mat = np.zeros((map_h, map_w))
        for g in self.generals:
            if g > 0:
                i = g // map_w
                j = g % map_w
                generals_mat[i, j] = 1
        self.board.static_game_layers[Board.INDEX_GENERALS] = generals_mat
        
        # armies
        for p in self.players:
            self.board.player_game_layers[p] = (terrain_mat >= 0) * (terrain_mat == p.get_index()) * np.array(armies).reshape((map_h, map_w))
        
        return self.board
        
    
#gm = GameManager()

#print(gm.patch([0, 0], [1, 1, 3]))
#print(gm.patch([0, 0], [0, 1, 2, 1]))
#print(gm.patch([], [0]))
#print(gm.patch([0, 686, 19, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -3, -3, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -4, -3, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -4, -4, -3, -3, -4, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -4, -3, -3, -4, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -4, -3, -4, -4, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -4, -4, -3, -3, -3, -3, -4, -4, -3, -3, -3, -4, -3, -3, -3, -4, -4, -4, -3, -3, -4, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -3, -3, -1, -1, -2, -3, -3, -3, -4, -3, -3, -3, -3, -4, -3, -3, -3, -4, -3, -4, -3, -2, 1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -4, -3, -3, -4, -3, -2, -1, -1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -4, -4], [308, 1, 2, 377]))