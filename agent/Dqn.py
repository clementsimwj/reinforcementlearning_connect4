from torch import nn

# Input dimension is 2 * 6 * 7 because we have two channels (one for each player) and a 6x7 board.
# Output dimension is 7 because there are 7 possible columns to place a piece in Connect Four.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    