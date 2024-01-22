'''
Created on Jun 24, 2023

@author: iapalm
'''

from connection_manager import ConnectionManager
from display_manager import console_display

from load_dotenv import load_dotenv

import logging
import os

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

## game presets
default_game_id = "ian_and_bot_match"
bot_username = "[Bot] Robocop"

print("--- Initializing Robocop control panel ---")
bot_id = os.getenv("BOT_ID")

# change display here, set to none for no cli display
#cm = ConnectionManager(bot_id, display=None)
cm = ConnectionManager(bot_id, display=console_display)
cm.do_connect()

do_continue = True
while do_continue:
    cmd = input(">> ")
    
    if cmd == "connect":
        cm.do_connect()
    elif cmd == "username":
        username = input("set username: ")
        cm.do_set_username(username)
    elif cmd == "disconnect":
        do_continue = False
        logging.info("disconnecting")
    elif cmd == "join":
        game_id = input("game id: ")
        
        cm.do_join_game(default_game_id if len(game_id) == 0 else game_id)
    elif cmd == "team":
        team_id = input("team id: ")
        
        cm.do_join_team(team_id)
    elif cmd == "start":
        cm.do_force_start(True)
    elif cmd == "cancel":
        cm.do_force_start(False)
    if cmd == "help":
        print("available commands: \n \
                >> connect: connect to server \n \
                >> username: set username \n \
                >> disconnect: disconnect from game\n \
                >> join: join game \n \
                >> start: force start \n \
                >> cancel: cancel force start \n \
                >> help: you seem to know what this does already \n \
                ")
        