'''
Created on Jun 22, 2023

@author: iapalm
'''

from game_manager import GameManager

import socketio
import logging


class ConnectionManager():
    def __init__(self, bot_id, display=None):
        self.bot_id = bot_id
        
        self.display = display
        
        sio = socketio.Client()
        
        self.sio = sio
        
        @sio.on("disconnect")
        def disconnect():
            self.on_disconnect()
            
        @sio.on("connect")
        def connect():
            self.on_connect()
            
        @sio.on("error_set_username")
        def error_set_username(error_str):
            self.on_error_set_username(error_str)
            
        @sio.on("game_start")
        def start(data, other):
            self.on_start(data, other)
            
        @sio.on("game_update")
        def game_update(data, other):
            move = self.gm.process_turn(data)
            
            if move is not None:
                self.do_attack(start=move["start"], end=move["end"], is50=move["is50"])
                
        @sio.on("game_lost")
        def game_lost(data, other):
            print("Defeated by {}".format(data["killer"]))
            
        @sio.on("game_won")
        def game_won(data, other):
            print("You won!")
            
    def on_disconnect(self):
        logging.warning("disconnected from server")
        
    def on_connect(self):
        logging.info("connected to server successfully")
            
    def on_error_set_username(self, error_str):
        if len(error_str) == 0:
            logging.info("successfully set username")
        else:
            logging.error(error_str)
    
    def on_start(self, data, other):
            player_index = data["playerIndex"]
            self.gm.set_player_index(player_index)
            
            usernames = data["usernames"]
            teams = data["teams"]
            self.gm.add_players(usernames, teams)
            
            replay_url = 'https://bot.generals.io/replays/' + data["replay_id"]
            logging.info("replay will be available after the game at {}".format(replay_url))
            
    def do_connect(self):
        self.sio.connect('https://bot.generals.io') # bot server has ssl cert issues :(
   
    def do_set_username(self, username):
        self.sio.emit('set_username', (self.bot_id, username))
    
    def do_join_game(self, game_id):
        self.game_id = game_id
        self.sio.emit('join_private', (game_id, self.bot_id))
        
        self.gm = GameManager(self.display)
    
        logging.info("https://bot.generals.io/games/{}".format(game_id))
        
    def do_attack(self, start, end, is50):
        self.sio.emit('attack', (start, end, is50))
    
    def do_join_team(self, team_id):
        if not hasattr(self, "game_id"):
            logging.error("you must join a game first")
        else:
            self.sio.emit('set_custon_team', (self.game_id, team_id))
            logging.debug("joined team {}".format(team_id))
    
    def do_force_start(self, force_start_status):
        if not hasattr(self, "game_id"):
            logging.error("you must join a game first")
        else:
            self.sio.emit('set_force_start', (self.game_id, force_start_status))
            logging.debug("force start set to {}".format(force_start_status))