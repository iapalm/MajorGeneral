'''
Created on Jun 25, 2023

@author: iapalm
'''

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

class Brain():
    def __init__(self, name):
        self.name = name
        self.defeated = False
        
    def set_index(self, index):
        self.index = index
        
    def get_index(self):
        return self.index
    
    def set_team(self, team):
        self.team = team
        
    def get_team(self):
        return self.team
        
    def __eq__(self, other):
        return self.name == other.name
    
    def set_defeated(self, is_defeated):
        self.defeated = is_defeated
        
    def is_defeated(self):
        return self.defeated
    
    def turn(self, state, metrics, fog_board):
        pass
    
    def reset(self):
        self.defeated = False
        self.team = None
        self.index = None
        
        return self
    
    def copy(self):
        b = Brain(self.name)
        b.defeated = self.defeated
        b.index = self.index
        b.team = self.team
        
        return b
    
    def __hash__(self):
        return hash(self.name)