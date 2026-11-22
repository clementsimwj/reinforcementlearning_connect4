from reinforcementlearning_connect4.game.utils import check_board_full, check_win, create_board, display_board, place_piece
from reinforcementlearning_connect4.game.config import config
import numpy as np

def main():
    board, turn = initialise_game()
    play_game(board, turn)

def play_game(board: np.ndarray, turn: int) -> None:
    players = ["player1", "player2"]
    has_winner = False
    while not has_winner:
        display_board(board)
        if check_board_full(board):
            print("The board is full! It's a draw.")
            return
        curr_player = players[turn % 2]

        col = get_player_move(board, curr_player)
        if col is None:
            print("Game Aborted.")
            return

        place_piece(board, col, curr_player)
        won = check_win(board, curr_player)
        if won:
            display_board(board)
            has_winner = True
            print(f"{config[curr_player]['name']} wins!")
            return
        turn += 1


def initialise_game() -> tuple[np.ndarray, int]:
    rows = config["rows"]
    cols = config["cols"]
    board = create_board(rows, cols)
    turn = 0
    return board, turn

def get_player_move(board: np.ndarray, player: str) -> int:
    name = config[player]["name"]
    rows, cols = board.shape
    while True:
        raw = input(f"{name}, choose a column (0-{cols-1}) or 'q' to quit: ").strip().lower()
        if raw == 'q':
            return None
        if not raw.isdigit():
            continue
        col = int(raw)
        if not 0 <= col < cols:
            continue
        if board[rows-1][col] != 0:
            print("Column is Full!")
            continue
        return col


if __name__ == "__main__":
    main()