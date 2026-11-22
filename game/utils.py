import numpy as np
from reinforcementlearning_connect4.game.config import config

# Board

def create_board(rows: int, cols: int) -> np.ndarray:
    """
    returns a 2D array of zeroes with the supplied number of rows and columns
    """
    board = np.zeros((rows, cols), dtype=int)
    return board

def display_board(board: np.ndarray) -> None:
    print(np.flip(board, 0))

# Actions

def place_piece(board:np.ndarray, col: int, player: str) -> None:
    size = board.shape
    symbol = config[player].get("symbol", 0)
    for row in range(size[0]):
        if board[row][col] == 0:
            board[row][col] = symbol
            
            return board

# Game Conditions

def check_board_full(board: np.ndarray) -> bool:
    return not np.any(board == 0)

def check_win(board: np.ndarray, player: str) -> bool:
    symbol = config[player].get("symbol", 0)
    (rows, cols) = board.shape

    # Check Horizontal:
    for row in range(rows):
        for col in range(cols - 3):
            if all(board[row][col+i] == symbol for i in range(4)):
                return True

    # Check Vertical:
    for col in range(cols):
        for row in range(rows - 3):
            if all(board[row+i][col] == symbol for i in range(4)):
                return True

    # Check Diagonal (bottom-left and top-right):
    for row in range(rows - 3):
        for col in range(cols - 3):
            if all (board[row+i][col+i] == symbol for i in range(4)):
                return True

    # Check Diagonal (top-left and bottom-right):
    for row in range(3, rows):
        for col in range(cols - 3):
            if all (board[row-i][col+i] == symbol for i in range(4)):
                return True

    return False


        
