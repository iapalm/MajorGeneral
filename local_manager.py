'''
Created on Jun 22, 2023

@author: iapalm
'''

from game_manager import GameManager
from board import Board
from random_brain import RandomBrain
from metrics_brain import MetricsBrain, MetricsMonteCarloBrain
from cnn_brain import CNNBrain, CNNMonteCarloBrain
from display_manager import console_display

import time
import os
import logging
from time import sleep

class LocalManager():
    def __init__(self, h, w, primary_bot, other_bots=[], teams=[], max_turns=1000, display=None, time_delay=0):
        assert len(other_bots) > 0
        self.primary_bot = primary_bot
        self.other_bots = other_bots
        self.max_turns = max_turns
        self.display = display
        self.time_delay = time_delay
        self.turn_number = 1
        
        players = []
        for i, b in enumerate([primary_bot] + other_bots):
            b.reset()
            b.set_index(i)
            if len(teams) > 0:
                b.set_team(teams[i])
            else:
                b.set_team(i)
            players.append(b)
        self.players = players
        
        self.board = Board(h, w, players=self.players)
        
    def play(self):
        end_game = False
        winners = None
        while not end_game and self.turn_number < self.max_turns:
            end_game, winners, _ = self.turn()
        if self.display is not None:
            print("game over, the bot {}".format("won" if (self.primary_bot in winners) else "lost"))
        return winners
    
    def turn(self):
        start_turn_time = time.time()
        end_game = False
        winners = set()
        self.turn_number += 1
        moves = [(p, p.turn(self.board.view_board(p, fog=True), self.board.get_metrics(p), self.board.fog_mask_copy(p))) for p in self.players]
        self.board.process_turn(moves, self.turn_number)
        
        if self.display is not None:
            os.system("cls")
            self.display(self.board.view_board(self.primary_bot, fog=True), fog=True)
            self.display(self.board.view_board(self.other_bots[0], fog=True), fog=True)
        
        remaining_players = {p for p in self.players if not p.is_defeated()}
        
        if self.turn_number > self.max_turns:
            end_game = True
        elif len({p.get_team() for p in remaining_players}) == 1:
            end_game = True
            
            winners = remaining_players
        
        end_turn_time = time.time()
        if self.time_delay > 0 and end_turn_time - start_turn_time > 0:
            sleep(end_turn_time - start_turn_time)
            
        return (end_game, winners, self.board.get_metrics(self.primary_bot))

if __name__ == "__main__":
    rb1 = CNNMonteCarloBrain(from_checkpoint="models/real_value_metrics/self-play-1/model_ep_best.pt", name="robocop")
    rb2 = MetricsBrain("skynet")
    
    # Other bot options
    #rb3 = MetricsMonteCarloBrain("terminator")
    #rb4 = CNNBrain(from_checkpoint="models/real_value_metrics/cnnmodel_32e_200g_initial.pt", name="terminator")
    #rb5 = CNNBrain(from_checkpoint="models/real_value_metrics/self-play-1/model_ep_best.pt", name="terminator")
    #rb6 = RandomBrain("irobot")
    
    # 4-player game with teams
    #lm = LocalManager(10, 10, primary_bot=rb1, other_bots=[rb2, rb3, rb4], teams=[1, 1, 2, 2], display=console_display)
    
    lm = LocalManager(8, 8, primary_bot=rb1, other_bots=[rb2], teams=[], display=console_display, time_delay=0.0)
    lm.play()