# Building a Machine Learning-Based generals.io Bot #
This repository is associated with the Numbers Game article at https://ianpalmer.io/numbersgame/blog/generals-io-part-1.

## Quick Setup ##
To run a local game of a CNNBrain with MCTS against a MetricsBrain, simply run `python local_manager.py`.

To play against a bot online, follow these steps:
1. Change the bot you want to play against in `game_manager.py`
2. `python cli_controller.py`
3. `join`
4. Enter a custom game id, or leave blank for the default
5. Visit the url printed in the console in a web browser to join the game
6. `start`
7. Press "Force Start" from the browser
8. Enjoy!

If you want to print the bot's view the console during the game, uncomment [this line](https://github.com/iapalm/MajorGeneral/blob/59f53e4b136d1378d97c614c6713f65b80a516e8/cli_controller.py#L28) and comment out the line before it.
