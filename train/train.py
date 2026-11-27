import random
import sys
from reinforcementlearning_connect4.game.utils import check_board_full, check_win, create_board, place_piece
from reinforcementlearning_connect4.game.config import config
from reinforcementlearning_connect4.agent.Agent import Agent, State

PLAYERS = ["player1", "player2"]

def play_training_game(agent: Agent) -> tuple[int, float]:
    board = create_board(config["rows"], config["cols"])
    turn = 0
    losses = []
    
    while True:
        name = PLAYERS[turn % 2]
        opp_name = PLAYERS[(turn + 1) % 2]
        symbol = config[name]["symbol"]
        opp_symbol = config[opp_name]["symbol"]
        
        state = State(board)
        s = state.encode_state(symbol)
        col = agent.select_action(state, symbol=symbol, training=True)
        
        place_piece(board, col, name)
        won = check_win(board, name)
        done = won or check_board_full(board)
        reward = 1.0 if won else 0.0
        
        next_state = State(board)
        s_next = next_state.encode_state(opp_symbol)
        agent.remember(s, col, reward, s_next, done, next_state.legal_mask())
        
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        turn+=1
        if done:
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            return turn, avg_loss
    
def play_random(agent: Agent, agent_name: str) -> str:
    board = create_board(config["rows"], config["cols"])
    turn = 0
    while True:
        name = PLAYERS[turn % 2]
        symbol = config[name]["symbol"]
        state = State(board)
        if name == agent_name:
            col = agent.select_action(state, symbol=symbol, training=False)
        else:
            col = random.choice(state.legal_columns())
        place_piece(board, col, name)
        if check_win(board, name):
            return "win" if name == agent_name else "loss"
        if check_board_full(board):
            return "draw"
        turn += 1

def evaluate(agent: Agent, n_games: int = 100) -> dict:
    results =  {"win": 0, "loss": 0, "draw": 0}
    for i in range(n_games):
        agent_name = PLAYERS[i%2]
        results[play_random(agent, agent_name)] += 1
    return results

def train(n_games: int=5000, eval_every: int = 500, eval_games: int = 100,
          eps_decay: float = 0.999, save_path: str = "dqn_weights.pt") -> Agent: 
    agent = Agent(symbol=config["player1"]["symbol"], epsilon_decay=eps_decay)
    recent_losses = []
    
    for game in range(1, n_games+1):
        _, avg_loss = play_training_game(agent)
        agent.decay_epsilon()
        recent_losses.append(avg_loss)
        
        if game % eval_every == 0:
            r = evaluate(agent, eval_games)
            print(f"game {game:>6} | eps {agent.epsilon:.3f} | "
                  f"avg loss {sum(recent_losses) / len(recent_losses):.4f} | "
                  f"vs random: {r['win']}W {r['loss']}L {r['draw']}D", flush=True)
            recent_losses = []
            agent.save(save_path)
            
    return agent
    
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    train(n_games=n, eval_every=max(1, n // 10), eps_decay=0.998 if n < 5000 else 0.999)
    
            
