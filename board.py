'''
Created on Aug 1, 2023

@author: iapalm
'''

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

class Board():
    TILE_EMPTY = -1
    TILE_MOUNTAIN = -2
    TILE_FOG = -3
    TILE_FOG_OBSTACLE = -4
    
    INDEX_EMPTY = 0
    INDEX_MOUNTAIN = 1
    INDEX_FOG = 2
    INDEX_FOG_OBSTACLE = 3
    INDEX_CITIES = 4
    INDEX_GENERALS = 5
    INDEX_SELF_ARMIES = 6
    INDEX_NEUTRAL_ARMIES = 7
    INDEX_ENEMY_ARMY_START = 8
    
    def __init__(self, dim=20, players=[]):
        self.dim = dim
        self.players = players
        self.player_indices = {players[i]: i for i in range(len(players))}
        self.initialize_board(players)
        
    def copy(self):
        board = Board(self.dim, self.players)
        board.dim = self.dim
        board.players = [x for x in self.players]
        board.player_indices = {k: v for k, v in self.player_indices.items()}
        board.player_game_layers = {k: v.copy() for k, v in self.player_game_layers.items()}
        board.player_fog_layers = {k: v.copy() for k, v in self.player_fog_layers.items()}
        board.static_game_layers = {k: v.copy() for k, v in self.static_game_layers.items()}
        
        return board
    
    def _apply_fog_mask(self, layer, fog_mask):
        return layer * (1 - fog_mask)
            
    def view_board(self, player, fog=True):
        board_view = np.zeros((12, self.dim, self.dim))
        apply_fog = self._apply_fog_mask if fog else (lambda x, y: x)
        
        fog_mask = self.player_fog_layers[player]
        board_view[self.INDEX_EMPTY, :, :] = apply_fog(self.static_game_layers[self.INDEX_EMPTY], fog_mask)
        board_view[self.INDEX_MOUNTAIN, :, :] = apply_fog(self.static_game_layers[self.INDEX_MOUNTAIN], fog_mask)
        board_view[self.INDEX_FOG, :, :] = fog_mask
        board_view[self.INDEX_FOG_OBSTACLE, :, :] = apply_fog(self.static_game_layers[self.INDEX_MOUNTAIN] + self.static_game_layers[self.INDEX_CITIES], 1 - fog_mask)
        board_view[self.INDEX_CITIES, :, :] = apply_fog(self.static_game_layers[self.INDEX_CITIES], fog_mask)
        board_view[self.INDEX_GENERALS, :, :] = apply_fog(self.static_game_layers[self.INDEX_GENERALS], fog_mask)
        board_view[self.INDEX_SELF_ARMIES, :, :] = self.player_game_layers[player]
        board_view[self.INDEX_NEUTRAL_ARMIES, :, :] = apply_fog(self.static_game_layers[self.INDEX_NEUTRAL_ARMIES], fog_mask)
        
        other_offset = 0
        for other, index in self.player_indices.items():
            if not other == player:
                if index - other_offset > 2:
                    board_view[self.INDEX_SELF_ARMIES + 5, :, :] += self.player_game_layers[other]
                else:
                    board_view[self.INDEX_ENEMY_ARMY_START + index - other_offset, :, :] = self.player_game_layers[other]
            else:
                other_offset = 1
                
        return board_view
    
    def modify_static_state(self, layer, i, j, val):
        self.static_game_layers[layer][i, j] = val
        
    def modify_player_state(self, player, i, j, val):
        self.player_game_layers[player][i, j] = val
        
    def modify_player_fog(self, player, i, j, val):
        self.player_fog_layers[player][i, j] = val
    
    def modify_player_fog_slice(self, player, slice, val):
        self.player_fog_layers[player][slice] = val    
    
    def get_tile_owner(self, player, layer_index):
        adjusted = layer_index - self.INDEX_ENEMY_ARMY_START
        player_offset = 1 if self.player_indices[player] <= adjusted else 0
        for p, i in self.player_indices.items():
            if i == adjusted + player_offset:
                return p
        raise ValueError()
    
    def process_turn(self, moves, reinforce_cities=False, reinforce_all=False):
        # increment generals and cities
        if reinforce_cities:
            for p in self.players:
                self.player_game_layers[p] += self.static_game_layers[self.INDEX_CITIES] * (self.player_game_layers[p] > 0)
                self.player_game_layers[p] += self.static_game_layers[self.INDEX_GENERALS] * (self.player_game_layers[p] > 0)
        
        # reinforce all tiles
        if reinforce_all:
            for p in self.players:
                self.player_game_layers[p] = (self.player_game_layers[p] + 1) * (self.player_game_layers[p] > 0)
        
        for p, m in moves:
            self.move(p, m)
        
        
    def move(self, player, move):
        if move is None:
            return
        # {"start": int(start_index), "end": int(target_index), "is50": False}
        def valid_coord(x):
            return (0 <= x) and (x < self.dim)
         
        start_i = int(move["start"] / self.dim)
        start_j = move["start"] % self.dim
        
        end_i = int(move["end"] / self.dim)
        end_j = move["end"] % self.dim
        
        game_state = self.view_board(player)
        
        if game_state[self.INDEX_SELF_ARMIES, start_i, start_j] > 1: # player owns tile and can make a move
            if valid_coord(start_i) and valid_coord(start_j) and valid_coord(end_i) and valid_coord(end_j): # coords are in bounds
                if abs(start_i - end_i) + abs(start_j - end_j) == 1: # can only move horiz/vert one tile
                    enemy_armies = game_state[self.INDEX_ENEMY_ARMY_START:, end_i, end_j]
                    self_start_armies = game_state[self.INDEX_SELF_ARMIES, start_i, start_j]
                    moving_armies = round(self_start_armies / (2 if move["is50"] else 1)) - 1
                    
                    if game_state[self.INDEX_EMPTY, end_i, end_j] == 1:
                        # destination is empty
                        self.modify_static_state(self.INDEX_EMPTY, end_i, end_j, 0)
                        
                        # move armies, checking if 50
                        self.modify_player_state(player, end_i, end_j, moving_armies)
                        # reduce armies at start
                        self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                        
                        # adjust fog mask
                        self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.dim)), range(max(end_j - 1, 0), min(end_j + 2, self.dim))), 0)
                    elif game_state[self.INDEX_MOUNTAIN, end_i, end_j] == 1:
                        # destination is mountain
                        pass
                    elif game_state[self.INDEX_NEUTRAL_ARMIES, end_i, end_j] > 0:
                        # destination is neutral army
                        enemy_army_count = game_state[self.INDEX_NEUTRAL_ARMIES, end_i, end_j]
                        
                        if moving_armies > enemy_army_count:
                            # destination results in defeat
                            # reduce own armies
                            self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                            
                            # reduce enemy armies to 0
                            self.modify_static_state(self.INDEX_NEUTRAL_ARMIES, end_i, end_j, 0)
                            # set own armies to remaining total
                            self.modify_player_state(player, end_i, end_j, moving_armies - enemy_army_count)
                            
                            # adjust fog of player
                            self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.dim)), range(max(end_j - 1, 0), min(end_j + 2, self.dim))), 0)
                        else:
                            # not a defeat
                            # reduce own armies
                            self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                            
                            # reduce enemy armies by moving units
                            self.modify_static_state(self.INDEX_NEUTRAL_ARMIES, end_i, end_j, enemy_army_count - moving_armies)
                    elif np.sum(enemy_armies) > 0: # destination is enemy
                        #print(enemy_armies)
                        enemy_army_index = np.where(enemy_armies > 0)[0][0] + self.INDEX_ENEMY_ARMY_START
                        enemy_army_count = game_state[enemy_army_index, end_i, end_j]
                        enemy_player = self.get_tile_owner(player, enemy_army_index)
                        #print("enemy army index {}".format(enemy_army_index))
                        #print("{} is attacking {} armies owned by {}".format(player.name, enemy_army_count, enemy_player.name))
                        
                        if moving_armies > enemy_army_count:
                            # destination results in defeat
                            # reduce own armies
                            self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                            
                            # reduce enemy armies to 0
                            self.modify_player_state(enemy_player, end_i, end_j, 0)
                            # set own armies to remaining total
                            self.modify_player_state(player, end_i, end_j, moving_armies - enemy_army_count)
                            
                            # does tile contain a general?
                            if game_state[self.INDEX_GENERALS, end_i, end_j] == 1:
                                # add all enemy armies to player
                                self.player_game_layers[player] += self.player_game_layers[enemy_player]
                                
                                # remove all enemy armies
                                self.player_game_layers[enemy_player] = np.zeros((self.dim, self.dim))
                                
                                # recalculate player fog
                                self.player_fog_layers[player] = self.player_fog_layers[player] * self.player_fog_layers[enemy_player]
                                
                                # defeat enemy player
                                enemy_player.set_defeated(True)
                                
                                # set tile to be a city instead of a general
                                self.modify_static_state(self.INDEX_GENERALS, end_i, end_j, 0)
                                self.modify_static_state(self.INDEX_CITIES, end_i, end_j, 1)
                            else:
                                # no general, standard defeat
                                # adjust fog of player
                                self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.dim)), range(max(end_j - 1, 0), min(end_j + 2, self.dim))), 0)
                                
                                # adjust fog of enemy, have to recalculate 3x3 square
                                #self.modify_player_fog_slice(enemy_player, np.ix_(range(0, self.dim), range(0, self.dim)), 1)
                                for i in range(max(end_i - 1, 0), min(end_i + 2, self.dim)):
                                    for j in range(max(end_j - 1, 0), min(end_j + 2, self.dim)):
                                        if np.sum(game_state[enemy_army_index, max(i - 1, 0): min(i + 2, self.dim), max(j - 1, 0): min(j + 2, self.dim)]) > 0:
                                            self.modify_player_fog(enemy_player, i, j, 0)
                                        else:
                                            self.modify_player_fog(enemy_player, i, j, 1)
                        else:
                            # not a defeat
                            # reduce own armies
                            self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                            
                            # reduce enemy armies by friendly count - 1
                            self.modify_player_state(enemy_player, end_i, end_j, enemy_army_count - moving_armies)
                            
                            if enemy_army_count == moving_armies:
                                self.modify_static_state(self.INDEX_EMPTY, end_i, end_j, 1)
                    elif game_state[self.INDEX_SELF_ARMIES, end_i, end_j] > 0:
                        # destination is own by self
                        self_start_armies = game_state[self.INDEX_SELF_ARMIES, start_i, start_j]
                        self_end_armies = game_state[self.INDEX_SELF_ARMIES, end_i, end_j]
                        moving_armies = round(self_start_armies / (2 if move["is50"] else 1)) - 1
                        
                        # reduce own armies in start
                        self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                        # add to armies at destination
                        self.modify_player_state(player, end_i, end_j, self_end_armies + moving_armies)
        
    def initialize_board(self, players):
        # not concerned with fog right now.  that will be generated for each player individually
        static_game_layers = {tile: np.zeros((self.dim, self.dim)) for tile in (self.INDEX_EMPTY, self.INDEX_MOUNTAIN, self.INDEX_FOG, self.INDEX_FOG_OBSTACLE, self.INDEX_CITIES, self.INDEX_GENERALS, self.INDEX_NEUTRAL_ARMIES)}
        player_game_layers = {player: np.zeros((self.dim, self.dim)) for player in players}
        player_fog_layers = {player: np.ones((self.dim, self.dim)) for player in players}
        
        static_game_layers[self.INDEX_EMPTY][:, :] = 1 # assume empty until we put something down
        
        # mountains comprise approx 15 percent of board
        mountains_placed = 0
        while mountains_placed < self.dim**2 * 0.15:
            x = np.random.randint(0, self.dim)
            y = np.random.randint(0, self.dim)
            static_game_layers[self.INDEX_MOUNTAIN][y, x] = 1
            static_game_layers[self.INDEX_EMPTY][y, x] = 0
            mountains_placed += 1
            
        # cities comprise approx 2.5 percent of board
        cities_placed = 0
        while cities_placed < self.dim**2 * 0.025:
            x = np.random.randint(0, self.dim)
            y = np.random.randint(0, self.dim)
            #if np.sum(static_game_layers[1:6, :, :], axis=0)[y, x] == 0: # check for existing conflicts
            if static_game_layers[self.INDEX_EMPTY][y, x] == 1: # check if tile is empty
                static_game_layers[self.INDEX_CITIES][y, x] = 1
                static_game_layers[self.INDEX_NEUTRAL_ARMIES][y, x] = np.random.randint(40, 50)
                static_game_layers[self.INDEX_EMPTY][y, x] = 0
                cities_placed += 1
                
        # set player starting positions
        #min_distance = dim - 2
        #circle_center = (int(self.dim / 2) + np.random.randint(-3, 4), int(self.dim / 2) + np.random.randint(-3, 4))
        circle_center = (int(self.dim / 2), int(self.dim / 2))
        circle_radius = (self.dim - 2) / 2
        player_coords = dict()
        
        theta = np.random.random() * 2 * np.pi
        for player in players:
            coords_empty = False
            coords_empty_tries = 0
            while not coords_empty:
                coords = [np.clip(int(circle_center[1] + circle_radius * np.sin(theta)) + np.random.randint(-2 - coords_empty_tries, 3 + coords_empty_tries), 1, self.dim - 1),
                                     np.clip(int(circle_center[0] + circle_radius * np.cos(theta)) + np.random.randint(-2 - coords_empty_tries, 3 + coords_empty_tries), 1, self.dim - 1)]
                if static_game_layers[self.INDEX_EMPTY][coords[0], coords[1]] == 1: # check if starting space empty
                    player_coords[player] = [x for x in coords]
                    coords_empty = True
                else:
                    coords_empty_tries += 1
            theta = (theta + (2 * np.pi / len(players))) % (2 * np.pi)
        
        for k, v in player_coords.items():
            # set the player value to 1
            player_game_layers[k][v[0], v[1]] = 1
            
            # adjust fog mask
            player_fog_layers[k][max(v[0] - 1, 0) : min(v[0] + 2, self.dim), max(v[1] - 1, 0) : min(v[1] + 2, self.dim)] = 0
            
            # adjust empty mask and general layer
            static_game_layers[self.INDEX_EMPTY][v[0], v[1]] = 0
            static_game_layers[self.INDEX_GENERALS][v[0], v[1]] = 1
        
        self.player_game_layers = player_game_layers
        self.player_fog_layers = player_fog_layers
        self.static_game_layers = static_game_layers