'''
Created on Jun 22, 2023

@author: iapalm
'''

from game_manager import GameManager
from board import Board
from random_brain import RandomBrain
from metrics_brain import MetricsBrain
from display_manager import console_display

import os
import logging
from time import sleep

class LocalManager():
    def __init__(self, dim, primary_bot, other_bots=[], display=None):
        assert len(other_bots) > 0
        self.primary_bot = primary_bot
        self.other_bots = other_bots
        self.display = display
        
        self.board = Board(dim=dim, players=[primary_bot] + other_bots)
    
    def sample_turn(self, player, move):
        new_board = self.board.copy()
        new_board.move(player, move)
        return new_board.view_board(player, fog=True)
        
    def play(self):
        end_game = False
        turn_number = 0
        while not end_game:
            os.system("cls")
            turn_number += 1
            moves = [(p, p.turn(self.board.view_board(p, fog=True), lambda move: self.sample_turn(p, move))) for p in self.board.players]
            self.board.process_turn(moves, turn_number % 2 == 0, turn_number % 10 == 0)
            #if turn_number % 1 == 0:
            if self.display is not None:
                self.display(self.board.view_board(self.primary_bot, fog=True), fog=True)
                self.display(self.board.view_board(self.other_bots[0], fog=True), fog=True, color_offset=1)
            
            defeated_players = sum([p.is_defeated() for p in self.board.players])
            
            if defeated_players == len(self.board.players) - 1:
                end_game = True
                
            sleep(.25)
        
        print("game over")

rb1 = MetricsBrain("robocop")
rb2 = MetricsBrain("terminator")
lm = LocalManager(dim=8, primary_bot=rb1, other_bots=[rb2], display=console_display)
lm.play()