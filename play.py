import sys

from reinforcementlearning_connect4.game.utils import (
    check_board_full, check_win, create_board, display_board, place_piece,
)
from reinforcementlearning_connect4.game.config import config
from reinforcementlearning_connect4.game.game import get_player_move
from reinforcementlearning_connect4.agent.Agent import Agent, State

PLAYERS = ["player1", "player2"]


def play(weights_path: str = "dqn_weights.pt", human_first: bool = True) -> None:
    human_name = "player1" if human_first else "player2"
    bot_name = "player2" if human_first else "player1"

    bot = Agent(symbol=config[bot_name]["symbol"])
    bot.load(weights_path)

    board = create_board(config["rows"], config["cols"])
    turn = 0
    print(f"You are {config[human_name]['name']} (symbol {config[human_name]['symbol']}). "
          f"The bot is {config[bot_name]['name']} (symbol {config[bot_name]['symbol']}).")

    while True:
        display_board(board)
        if check_board_full(board):
            print("The board is full! It's a draw.")
            return

        name = PLAYERS[turn % 2]
        if name == human_name:
            col = get_player_move(board, name)
            if col is None:
                print("Game aborted.")
                return
        else:
            col = bot.make_move(State(board))
            print(f"Bot plays column {col}")

        place_piece(board, col, name)
        if check_win(board, name):
            display_board(board)
            print("You win!" if name == human_name else "The bot wins!")
            return
        turn += 1


if __name__ == "__main__":
    # python -m reinforcementlearning_connect4.play            -> you go first
    # python -m reinforcementlearning_connect4.play second     -> bot goes first
    play(human_first=not (len(sys.argv) > 1 and sys.argv[1] == "second"))