import random

class Agent:
    def __init__(self, symbol):
        self.symbol = symbol

    def make_move(self, state):
        return random.choice([col for col in range(state.board.shape[1]) if state.board[state.board.shape[0]-1][col] == 0])


class State:
    def __init__(self, board):
        self.board = board