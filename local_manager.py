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
    def __init__(self, h, w, primary_bot, other_bots=[], teams=[], display=None):
        assert len(other_bots) > 0
        self.primary_bot = primary_bot
        self.other_bots = other_bots
        self.display = display
        
        players = []
        for i, b in enumerate([primary_bot] + other_bots):
            b.set_index(i)
            if len(teams) > 0:
                b.set_team(teams[i])
            else:
                b.set_team(i)
            players.append(b)
        self.players = players
        
        self.board = Board(h, w, players=self.players)
    
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
            moves = [(p, p.turn(self.board.view_board(p, fog=True), lambda move: self.sample_turn(p, move))) for p in self.players]
            self.board.process_turn(moves, turn_number % 2 == 0, turn_number % 10 == 0)
            #if turn_number % 1 == 0:
            if self.display is not None:
                self.display(self.board.view_board(self.primary_bot, fog=True), fog=True)
                self.display(self.board.view_board(self.other_bots[0], fog=True), fog=True)
                #self.display(self.board.view_board(self.other_bots[1], fog=True), fog=True)
                #self.display(self.board.view_board(self.other_bots[2], fog=True), fog=True)
            
            remaining_players = [p for p in self.players if not p.is_defeated()]
            
            if len({p.get_team() for p in remaining_players}) == 1:
                end_game = True
                
            sleep(.5)
        
        print("game over")

rb1 = MetricsBrain("robocop")
rb2 = MetricsBrain("terminator")
rb3 = MetricsBrain("skynet")
rb4 = MetricsBrain("irobot")
lm = LocalManager(10, 10, primary_bot=rb1, other_bots=[rb2, rb3, rb4], teams=[1, 1, 2, 2], display=console_display)
lm.play()