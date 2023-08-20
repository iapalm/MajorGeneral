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
# layer 8: friendly armies (number representing quantity)
# layer 9-10: other player armies (number representing quantity)
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
    INDEX_FRIENDLY_ARMIES = 8
    INDEX_ENEMY_ARMY_START = 9
    
    def __init__(self, h, w, players=[], generate=True):
        self.h = h
        self.w = w
        self.players = players
                
        self.initialize_board(players, generate=generate)
            
    def from_game_state(self, state):
#         bot_name = "self"
#         self.player_indices = {0: bot_name}
#         for i in range(4):
#             if np.sum(state[8 + i, :, :]) > 0:
#                 self.player_indices[i] = str(i)
#         
#         self.player_fog_layers = {}
#         
#         self.player_fog_layers[bot_name] = state[2]
        raise NotImplementedError()
        
    def copy(self):
        board = Board(self.h, self.w, self.players)
        board.h = self.h
        board.w = self.w
        board.players = [x for x in self.players]
        board.player_game_layers = {k: v.copy() for k, v in self.player_game_layers.items()}
        board.player_fog_layers = {k: v.copy() for k, v in self.player_fog_layers.items()}
        board.static_game_layers = {k: v.copy() for k, v in self.static_game_layers.items()}
        
        return board
    
    def _apply_fog_mask(self, layer, fog_mask):
        return layer * (1 - fog_mask)
            
    def view_board(self, player, fog=True):
        board_view = np.zeros((12, self.h, self.w))
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
        
        
        placed_enemies = 0
        # enemies
        for other in sorted([p for p in self.players if p.get_team() != player.get_team()], key=lambda l: l.get_index()):
            board_view[self.INDEX_ENEMY_ARMY_START + min(placed_enemies, 2), :, :] = self.player_game_layers[other]
            placed_enemies += 1
        
        # friends
        for other in [p for p in self.players if p.get_team() == player.get_team()]:
            if not other == player:
                board_view[self.INDEX_FRIENDLY_ARMIES, :, :] += self.player_game_layers[other]
                
        return board_view
    
    def modify_static_state(self, layer, i, j, val):
        self.static_game_layers[layer][i, j] = val
        
    def modify_player_state(self, player, i, j, val):
        self.player_game_layers[player][i, j] = val
        
    def modify_player_fog(self, player, i, j, val):
        for p in self.players:
            if p.get_team() == player.get_team():
                self.player_fog_layers[p][i, j] = val
    
    def modify_player_fog_slice(self, player, slice, val):
        for p in self.players:
            if p.get_team() == player.get_team():
                self.player_fog_layers[p][slice] = val    
    
    def get_tile_owner(self, i, j):
        owners = []
        for p in self.players:
            if self.player_game_layers[p][i, j] > 0:
                owners.append(p)
        if len(owners) == 0:
            return None
        if len(owners) > 1:
            raise ValueError("The following players all own {}, {}: {}".format(i, j, owners))
        return owners[0]
    
    def get_friendly_owner(self, player, i, j):
        for p in self.players:
            if p.get_team() == player.get_team() and not p == player and self.player_game_layers[p][i, j] > 0:
                return p
        raise ValueError("No valid friendly player")
    
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
        def valid_coord_x(x):
            return (0 <= x) and (x < self.w)
        
        def valid_coord_y(y):
            return (0 <= y) and (y < self.h)
         
        start_i = int(move["start"] / self.w)
        start_j = move["start"] % self.w
        
        end_i = int(move["end"] / self.w)
        end_j = move["end"] % self.w
        
        game_state = self.view_board(player)
        
        if game_state[self.INDEX_SELF_ARMIES, start_i, start_j] > 1: # player owns tile and can make a move
            if valid_coord_y(start_i) and valid_coord_x(start_j) and valid_coord_y(end_i) and valid_coord_x(end_j): # coords are in bounds
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
                        self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.h)), range(max(end_j - 1, 0), min(end_j + 2, self.w))), 0)
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
                            self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.h)), range(max(end_j - 1, 0), min(end_j + 2, self.w))), 0)
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
                        enemy_player = self.get_tile_owner(end_i, end_j)
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
                                self.player_game_layers[enemy_player] = np.zeros((self.h, self.w))
                                
                                # recalculate player fog
                                for p in self.players:
                                    if p.get_team() == player.get_team():
                                        self.player_fog_layers[p] = self.player_fog_layers[p] * self.player_fog_layers[enemy_player]
                                
                                # defeat enemy player
                                enemy_player.set_defeated(True)
                                
                                # set tile to be a city instead of a general
                                self.modify_static_state(self.INDEX_GENERALS, end_i, end_j, 0)
                                self.modify_static_state(self.INDEX_CITIES, end_i, end_j, 1)
                            else:
                                # no general, standard defeat
                                # adjust fog of player
                                self.modify_player_fog_slice(player, np.ix_(range(max(end_i - 1, 0), min(end_i + 2, self.h)), range(max(end_j - 1, 0), min(end_j + 2, self.w))), 0)
                                
                                # adjust fog of enemy, have to recalculate 3x3 square
                                #self.modify_player_fog_slice(enemy_player, np.ix_(range(0, self.dim), range(0, self.dim)), 1)
                                for i in range(max(end_i - 1, 0), min(end_i + 2, self.h)):
                                    for j in range(max(end_j - 1, 0), min(end_j + 2, self.w)):
                                        if np.sum(game_state[enemy_army_index, max(i - 1, 0): min(i + 2, self.h), max(j - 1, 0): min(j + 2, self.w)]) > 0:
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
                    elif game_state[self.INDEX_FRIENDLY_ARMIES, end_i, end_j] > 0:
                        # destination is owned by a friend
                        self_start_armies = game_state[self.INDEX_SELF_ARMIES, start_i, start_j]
                        friend_end_armies = game_state[self.INDEX_FRIENDLY_ARMIES, end_i, end_j]
                        friend_player = self.get_tile_owner(end_i, end_j)
                        moving_armies = round(self_start_armies / (2 if move["is50"] else 1)) - 1
                        
                        # reduce own armies in start
                        self.modify_player_state(player, start_i, start_j, self_start_armies - moving_armies)
                        # add to armies at destination
                        self.modify_player_state(friend_player, end_i, end_j, friend_end_armies + moving_armies)
        
    def initialize_board(self, players, generate=True):
        # not concerned with fog right now.  that will be generated for each player individually
        static_game_layers = {tile: np.zeros((self.h, self.w)) for tile in (self.INDEX_EMPTY, self.INDEX_MOUNTAIN, self.INDEX_CITIES, self.INDEX_GENERALS, self.INDEX_NEUTRAL_ARMIES)}
        player_game_layers = {player: np.zeros((self.h, self.w)) for player in players}
        player_fog_layers = {player: np.ones((self.h, self.w)) for player in players}
        
        if generate == False:
            self.player_game_layers = player_game_layers
            self.player_fog_layers = player_fog_layers
            self.static_game_layers = static_game_layers
            
            return self
        
        static_game_layers[self.INDEX_EMPTY][:, :] = 1 # assume empty until we put something down
        
        # mountains comprise approx 15 percent of board
        mountains_placed = 0
        while mountains_placed < self.h * self.w * 0.15:
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            static_game_layers[self.INDEX_MOUNTAIN][y, x] = 1
            static_game_layers[self.INDEX_EMPTY][y, x] = 0
            mountains_placed += 1
            
        # cities comprise approx 2.5 percent of board
        cities_placed = 0
        while cities_placed < self.h* self.w * 0.025:
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            #if np.sum(static_game_layers[1:6, :, :], axis=0)[y, x] == 0: # check for existing conflicts
            if static_game_layers[self.INDEX_EMPTY][y, x] == 1: # check if tile is empty
                static_game_layers[self.INDEX_CITIES][y, x] = 1
                static_game_layers[self.INDEX_NEUTRAL_ARMIES][y, x] = np.random.randint(40, 50)
                static_game_layers[self.INDEX_EMPTY][y, x] = 0
                cities_placed += 1
                
        # set player starting positions
        #min_distance = dim - 2
        #circle_center = (int(self.dim / 2) + np.random.randint(-3, 4), int(self.dim / 2) + np.random.randint(-3, 4))
        circle_center = (int(self.h / 2), int(self.w / 2))
        circle_radius = (self.h - 2) / 2
        player_coords = dict()
        
        theta = np.random.random() * 2 * np.pi
        for player in players:
            coords_empty = False
            coords_empty_tries = 0
            while not coords_empty:
                coords = [np.clip(int(circle_center[1] + circle_radius * np.sin(theta)) + np.random.randint(-2 - coords_empty_tries, 3 + coords_empty_tries), 1, self.h - 1),
                                     np.clip(int(circle_center[0] + circle_radius * np.cos(theta)) + np.random.randint(-2 - coords_empty_tries, 3 + coords_empty_tries), 1, self.w - 1)]
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
            for p in players:
                if p.get_team() == k.get_team():
                    player_fog_layers[p][max(v[0] - 1, 0) : min(v[0] + 2, self.h), max(v[1] - 1, 0) : min(v[1] + 2, self.w)] = 0
            
            # adjust empty mask and general layer
            static_game_layers[self.INDEX_EMPTY][v[0], v[1]] = 0
            static_game_layers[self.INDEX_GENERALS][v[0], v[1]] = 1
        
        self.player_game_layers = player_game_layers
        self.player_fog_layers = player_fog_layers
        self.static_game_layers = static_game_layers