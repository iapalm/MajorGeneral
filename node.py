'''
Created on Sep 16, 2023

@author: iapalm
'''

import numpy as np

rng = np.random.default_rng()

class Node():
    def __init__(self, move, resulting_board, p_win, parent=None, c=1):
        self.move = move
        self.resulting_board = resulting_board
        self.p_win = p_win
        self.children = []
        self.N = 1
        self.c = c
        
        self.explored = False
        self.best_child = None
        self.best_child_p_win = None
        
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
        
    def get_best_child(self):
        if len(self.children) == 0 or self.p_win > self.best_child_p_win:
            return self
        else:
            return self.best_child
        
    def get_ucb_score(self, t):
        score = self.p_win
        if self.best_child_p_win is not None:
            score = max((self.p_win, self.best_child_p_win))
        
        parent_visits = t
        if self.parent is not None:
            parent_visits = self.parent.N
        
        return score + self.c * np.sqrt((2 * np.log(parent_visits)) / self.N)
        
    def select_to_explore(self, t):
        if len(self.children) == 0:
            if self.explored:
                return None
            else:
                return self
        else:
            for n in sorted(self.children, key=lambda c: c.get_ucb_score(t), reverse=True):
                result = n.select_to_explore(t)
                
                if result is not None:
                    return result
            return None
    
    def add_child(self, child):
        self.children.append(child)
        self.update(child, child.p_win)
            
    def update(self, child, p_new):
        if self.best_child_p_win is None or p_new > self.best_child_p_win:
            self.best_child = child
            self.best_child_p_win = p_new
            
            if self.parent is not None:
                self.parent.update(self, p_new)
                
    def visit(self):
        self.explored = True
        self.N += 1
        
        if self.parent is not None:
            self.parent.visit()
        
    def __len__(self):
        return 1 + sum([len(c) for c in self.children])
    
    def height(self):
        return 1 + max([0] + [c.height() for c in self.children])
    
    def depth(self):
        if self.parent is None:
            return 1
        else:
            return 1 + self.parent.depth()
    
    def __repr__(self):
        return "Node: {}: {} with {} children with max value {}, visited {} times, explored: {}:\n".format(self.move, self.p_win, len(self.children), self.best_child_p_win, self.N, self.explored)
    
    def print_tree(self, max_depth=None):
        if max_depth is None:
            max_depth = self.depth()
        rep = "- " + str(self)
        if max_depth > 0:
            for c in self.children:
                for l in c.print_tree(max_depth - 1).split("\n"):
                    rep += " " + l+"\n"
        return rep