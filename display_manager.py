'''
Created on Aug 5, 2023

@author: iapalm
'''

import numpy as np
import os
from colorama import init, Fore, Back, Style
from colorama.initialise import just_fix_windows_console
just_fix_windows_console()

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
    
COLOR_ORDER = [Back.GREEN, Back.RED, Back.BLUE, Back.YELLOW, Back.YELLOW]

def get_color(index):
    return COLOR_ORDER[min(index, len(COLOR_ORDER) - 1)]

def console_display(game_state, fog=False, color_offset=0):
    h = game_state.shape[1]
    w = game_state.shape[2]
    board_elements = [[(Back.BLACK, Fore.WHITE, "") for _ in range(w)] for _ in range(h)]
        
    for i in range(h):
        for j in range(w):
            back_color = Back.BLACK
            fore_color = Fore.WHITE
            tile_char = " "
            if fog and game_state[INDEX_FOG][i, j] == 1: # fog
                back_color = Back.LIGHTBLACK_EX
                tile_char = "F"
            elif fog and game_state[INDEX_FOG_OBSTACLE][i, j] == 1: # fog obstacle
                back_color = Back.LIGHTBLACK_EX
                tile_char = "O"
            elif game_state[INDEX_EMPTY][i, j] == 1: # empty
                back_color = Back.LIGHTBLACK_EX
                tile_char = "_"
            elif game_state[INDEX_MOUNTAIN][i, j] == 1: # mountain
                back_color = Back.LIGHTBLACK_EX
                tile_char = "M"
            else: # owned by a player
                if game_state[INDEX_SELF_ARMIES][i, j] > 0: # own army
                    tile_char = int(game_state[INDEX_SELF_ARMIES][i, j])
                    back_color = get_color(0 + color_offset)
                elif game_state[INDEX_NEUTRAL_ARMIES][i, j] > 0: # neutral army
                    tile_char = int(game_state[INDEX_NEUTRAL_ARMIES][i, j])
                    back_color = get_color(-1)
                elif game_state[INDEX_FRIENDLY_ARMIES][i, j] > 0: # friendly army
                    tile_char = int(game_state[INDEX_FRIENDLY_ARMIES][i, j])
                    back_color = Back.CYAN
                elif game_state[INDEX_ENEMY_ARMY_START][i, j] > 0: # enemy army 1
                    tile_char = int(game_state[INDEX_ENEMY_ARMY_START][i, j])
                    back_color = get_color(1 + color_offset)
                elif game_state[INDEX_ENEMY_ARMY_START + 1][i, j] > 0: # enemy army 2
                    tile_char = int(game_state[INDEX_ENEMY_ARMY_START + 1][i, j])
                    back_color = get_color(2 + color_offset)
                elif game_state[INDEX_ENEMY_ARMY_START + 2][i, j] > 0: # all other armies
                    tile_char = int(game_state[INDEX_ENEMY_ARMY_START + 2][i, j])
                    back_color = get_color(3 + color_offset)
                    
                if game_state[INDEX_GENERALS][i, j] > 0: # is a general
                    fore_color = Fore.YELLOW
                elif game_state[INDEX_CITIES][i, j] > 0: # is a city
                    fore_color = Fore.BLUE
                    
                    
            board_elements[i][j] = (back_color, fore_color, tile_char)
    
    #os.system("cls")
    for i in range(h):
        empty_row_str = ""
        num_row_str = ""
        for j in range(w):
            empty_row_str += board_elements[i][j][0] + board_elements[i][j][1] + "   "
            num_row_str += board_elements[i][j][0] + board_elements[i][j][1] + str(board_elements[i][j][2]).center(3)
        
        empty_row_str += Fore.WHITE + Back.BLACK
        num_row_str += Fore.WHITE + Back.BLACK
        print(empty_row_str)
        print(num_row_str)
        print(empty_row_str)
        
    print(Fore.WHITE + Back.BLACK)