import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as f
import copy
from Dqn import DQN

class Agent:
    def __init__(self, symbol: int, rows: int = 6, cols: int = 7,
                 gamma: float = 0.99, 
                 epsilon: float = 0.1, 
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 buffer_size: int = 100000,
                 target_sync_every: int = 500,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01):
        self.symbol = symbol
        self.actions = list(range(cols))  # There are 7 columns in Connect Four
        self.dqn = DQN(input_dim=2 * rows * cols, output_dim=cols)  # Assuming a rows x cols Connect Four board
        
        #hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_sync_every = target_sync_every
        self.learn_steps = 0
        self.buffer = deque(maxlen=buffer_size)
        
        self.target_dqn = copy.deepcopy(self.dqn)
        self.target_dqn.eval()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        
    def select_action(self, state: 'State', symbol: int = None, training: bool = False) -> int:
        if symbol is None:
            symbol = self.symbol
        legal_columns = state.legal_columns()
        if not legal_columns:
            raise ValueError("No legal moves available.")
        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_columns)
        x = torch.from_numpy(state.encode_state(symbol)).unsqueeze(0)
        with torch.no_grad():
            scores = self.dqn(x).squeeze(0)
        scores = scores.masked_fill(torch.from_numpy(~state.legal_mask()), float("-inf"))
        return int(torch.argmax(scores).item())
    
    def make_move(self, state: 'State') -> int:
        return self.select_action(state, training=False)
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def remember(self, s, a, r, s_next, done, next_legal_mask):
        """_summary_
        s: 84-number input state before the move (mover's pov)
        a: column played (0-6)
        r: reward (+1 for a winning move, else 0)
        s_next: 84-number state after the move, from the OPPONENT's pov
        done: True if game ended with this move
        next_legal_mask: 7 booleans, which columns can opponent play in s_next
        """
        self.buffer.append((s, a, r, s_next, done, next_legal_mask))
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s,a,r,s_next,done,mask = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32)
        s_next = torch.tensor(np.array(s_next), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        mask = torch.tensor(np.array(mask), dtype=torch.bool)
        
        q_pred = self.dqn(s).gather(1,a).squeeze(1)
        
        with torch.no_grad():
            q_next = self.target_dqn(s_next).masked_fill(~mask, float("-inf"))
            best_opp = q_next.max(dim=1).values
            best_opp = torch.where(done, torch.zeros_like(best_opp), best_opp)
            target = r - self.gamma * best_opp
            
        loss = f.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        self.learn_steps += 1
        if self.learn_steps % self.target_sync_every == 0:
            self.sync_target()
        return float(loss.item())
        
    def sync_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def save(self, path):
        torch.save(self.dqn.state_dict(), path)
    
    def load(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.sync_target()


class State:
    def __init__(self, board):
        self.board = board
        
    def encode_state(self, symbol: int) -> np.ndarray:
        curr_player_channel = self.board == symbol
        opponent_channel = (self.board != 0) & (self.board != symbol)
        return np.concatenate([curr_player_channel.flatten(), opponent_channel.flatten()]).astype(np.float32)
    
    def legal_mask(self):
        top_row = self.board.shape[0] - 1
        return self.board[top_row] == 0
    
    def legal_columns(self):
        return [int(col) for col in np.flatnonzero(self.legal_mask())]
    
    
if __name__ == "__main__":
    board = np.array([[1, 2, 1],
                      [0, 0, 0]])
    s = State(board)
    print("player 1 view:", s.encode_state(1))
    print("player 2 view:", s.encode_state(2))
    agent = Agent(symbol=1, batch_size=32, target_sync_every=50)
    board = np.zeros((6, 7), dtype=int)
    board[0, :3] = 1
    board[1, :3] = 2
    st = State(board)
    vec = st.encode_state(1)
    after = board.copy()
    after[0, 3] = 1
    vec_next = State(after).encode_state(2)
    full_mask = np.ones(7, dtype=bool)

    for _ in range(200):
        agent.remember(vec, 3, 1.0, vec_next, True, full_mask)   # winning move
        agent.remember(vec, 6, 0.0, vec_next, False, full_mask)  # a non-winning move
    for _ in range(400):
        agent.learn()

    with torch.no_grad():
        q = agent.dqn(torch.from_numpy(vec).unsqueeze(0)).squeeze(0)
    print("Q-values after training:", [round(float(x), 2) for x in q])
    print("greedy move:", agent.select_action(st, training=False), "(expected 3)")
    assert agent.select_action(st, training=False) == 3
    print("OK: agent learned the winning column")