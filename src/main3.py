from ttt.tic_tac_toe_board import TicTacToeBoard
from ttt.tic_tac_toe import TicTacToeTerminal, TicTacToeHeadless
from ttt.players import Player, RandomPlayer, OptimalPlayer
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, player=None, prob=1.):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.N = 0
        self.W = 0
        self.Q = 0
        self.c = .5
        self.P = prob
        self.depth = 0 if parent is None else parent.depth + 1

        if player:
            self.player = player
        else:
            self.player = -self.parent.player

    # trasverse tree until leaf node and return it
    def select(self):
        current_node = self
        while current_node.children:
            current_node = max(
                current_node.children, 
                key=lambda node: node.Q + self.c * self.P * np.sqrt(node.parent.N) / (1 + node.N) # add self.P when implemented
            )
        return current_node

    def expand(self, probs):
        valid_moves = self.game_state.get_valid_moves()
        for i, move in enumerate(valid_moves):
            next_state = self.game_state.make_move(move, self.player)
            child_node = MCTSNode(next_state, parent=self, move=move, prob=probs[i])
            self.children.append(child_node)

    # Evaluate position (through random play for example (like rollout policy))
    def evaluate(self, network):
        s = convert_state_to_tensor(self.game_state.board, player_perpective=self.player)
        P, v = network(s.unsqueeze(0).float())
        #return P.squeeze().detach().numpy(), v.item()

        current_state = self.game_state
        current_player = self.player
        # change this to set v to the output of the neural net
        while not current_state.is_game_over():
            possible_moves = current_state.get_valid_moves()
            move = random.choice(possible_moves)
            current_state = current_state.make_move(move, current_player)
            current_player *= -1

        #self.P = P # change this to be set to the output of the neural net
        return P.squeeze().detach().numpy(), current_state.get_game_result()
        # return v # --> just returning v like this is wrong because I'm returning v from
        # the perspective of the player who just moved but I need to return v from the
        # global perspective. the model is trained to return v from the perspective of the player
        # who is to move next. so I need to return v * self.player. I think. Check this
    #-(-1 * -1) * 1
    
    def backpropagate(self, result):
        current_node = self
        while current_node is not None:
            current_node.N += 1
            # the - here is needed because the result is from the perspective of the
            # player who just moved which is the parent of the current node
            current_node.W += -result * current_node.player
            current_node.Q = current_node.W / current_node.N
            current_node = current_node.parent

    def find_child_by_move(self, move):
        # Search through the children nodes for a node that corresponds to the given move.
        # Returns the child node if found, otherwise returns None.

        for child in self.children:
            if child.move == move:
                return child
        return None
    
    def find_child_by_state(self, game_state):
        # Search through the children nodes for a node that corresponds to the given game state.
        # Returns the child node if found, otherwise returns None.
        
        for child in self.children:
            if np.allclose(child.game_state.board, game_state.board):
                return child
        return None

    def mcts(self, iterations, network):
        for _ in range(iterations):
            node = self.select()
            if not node.game_state.is_game_over():
                probs, _ = node.evaluate(network=network)  # Get probabilities here
                node.expand(probs)  # Pass probabilities to expand method

            result = node.evaluate(network=network)[1]  # Only get the value here
            node.backpropagate(result)

    def get_best_child(self):
        # Returning the best move from this node's children
        return max(self.children, key=lambda node: node.N)
    
    def get_Ns(self):
        Ns = np.zeros((3, 3))
        for node in self.children:
            Ns[node.move] = node.N
        return Ns

    def get_Qs(self):
        Qs = np.zeros((3, 3))
        for node in self.children:
            Qs[node.move] = node.Q
        return Qs
 

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        # Adjust the first convolutional layer to accept 2 input channels
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)

        # Output layers
        # Move probabilities
        self.fc_probs = nn.Linear(32, 9)
        # Value estimation
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 3 * 3)

        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Apply the final layers for move probabilities and value estimation
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value
    
class SimpleTicTacToeNet(nn.Module):
    def __init__(self):
        super(SimpleTicTacToeNet, self).__init__()
        # Use only one convolutional layer
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)

        # Simplify the fully connected layers
        self.fc1 = nn.Linear(16 * 3 * 3, 32)

        # Output layers
        self.fc_probs = nn.Linear(32, 9)
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value
    
class TicTacToeNet2(nn.Module):
    def __init__(self):
        super(TicTacToeNet2, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability

        # Output layers
        self.fc_probs = nn.Linear(32, 9)
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 3 * 3)

        # Apply fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        # Apply the final layers for move probabilities and value estimation
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value
    
class ConvolutionalTicTacToeNetWithDropout(nn.Module):
    def __init__(self):
        super(ConvolutionalTicTacToeNetWithDropout, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Flattening layer for transition from convolutional layers to linear layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)

        # Output layers
        self.fc_probs = nn.Linear(32, 9)  # For 9 positions on the Tic-Tac-Toe board
        self.fc_value = nn.Linear(32, 1)  # For value estimation

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Apply dropout
        x = self.dropout(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layers
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value


    
class EnhancedTicTacToeNet(nn.Module):
    def __init__(self):
        super(EnhancedTicTacToeNet, self).__init__()
        # One convolutional layer
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)

        # More fully connected layers
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)

        # Output layers
        self.fc_probs = nn.Linear(32, 9)
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x):
        # Convolutional layer
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Output layers
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value

class LinearTicTacToeNet(nn.Module):
    def __init__(self):
        super(LinearTicTacToeNet, self).__init__()

        # Define the linear layers
        # Assuming a 3x3 Tic-Tac-Toe board with 2 channels, the flattened input size will be 3*3*2 = 18
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)

        # Output layers
        self.fc_probs = nn.Linear(32, 9)  # For 9 positions on the Tic-Tac-Toe board
        self.fc_value = nn.Linear(32, 1)  # For value estimation

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Output layers
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the shortcut to match the dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualTicTacToeNet(nn.Module):
    def __init__(self):
        super(ResidualTicTacToeNet, self).__init__()
        self.in_channels = 32

        # First layer with 5x5 kernel size
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks with 5x5 kernel size
        self.layer1 = ResidualBlock(32, 64)
        self.layer2 = ResidualBlock(64, 64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)

        # Output layers
        self.fc_probs = nn.Linear(32, 9)
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        probs = F.softmax(self.fc_probs(out), dim=1)
        value = torch.tanh(self.fc_value(out))
        return probs, value
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the shortcut to match the dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualTicTacToeNet(nn.Module):
    def __init__(self):
        super(ResidualTicTacToeNet, self).__init__()
        self.in_channels = 32

        # First layer with 5x5 kernel size
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks with 5x5 kernel size
        self.layer1 = ResidualBlock(32, 64)
        self.layer2 = ResidualBlock(64, 64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)

        # Output layers
        self.fc_probs = nn.Linear(64, 9)
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        probs = F.softmax(self.fc_probs(out), dim=1)
        value = torch.tanh(self.fc_value(out))
        return probs, value


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out += self.shortcut(x)
        return F.relu(out)

class SimpleResidualTicTacToeNet(nn.Module):
    def __init__(self):
        super(SimpleResidualTicTacToeNet, self).__init__()

        # First convolutional layer with 5x5 kernel size
        self.conv = nn.Conv2d(2, 16, kernel_size=5, padding=2)

        # One simple residual block
        self.res_block = SimpleResidualBlock(16, 16)

        # Intermediary fully connected layer
        self.inter_fc = nn.Linear(16 * 3 * 3, 32)

        # Fully connected layers for probabilities and value
        self.fc_probs = nn.Linear(32, 9)  # Outputting probabilities for 9 positions
        self.fc_value = nn.Linear(32, 1)  # Outputting value estimation

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.res_block(out)
        out = out.view(out.size(0), -1)

        # Intermediary layer
        out = F.relu(self.inter_fc(out))

        # Calculating probabilities and value
        probs = F.softmax(self.fc_probs(out), dim=1)
        value = torch.tanh(self.fc_value(out))
        return probs, value


# # Create an instance of the network
# net = TicTacToeNet()

# board = TicTacToeBoard()
# # Modified to pass tuples
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((0, 2), 1)
# s_raw = torch.tensor(board.board)

def convert_state_to_tensor(s_raw, player_perpective=1):
    s = torch.zeros((2, 3, 3), dtype=torch.int32)
    s[0, s_raw == player_perpective] = 1
    s[1, s_raw == -player_perpective] = 1
    return s

# # Usage
# s = convert_state_to_tensor(s_raw, player_perpective=-1)
# print(s_raw)
# print(s)

# P, v = net(s.unsqueeze(0).float())
# print(P)
# print(v)

class BetaZeroPlayer(Player):
    def __init__(self, player, model, verbose=False):
        super().__init__()
        self.player = player
        self.root_node = None
        self.simulations_per_turn = 25
        self.verbose = verbose
        self.probs_list = []
        self.model = model

    def get_move(self, board):
        if self.root_node is None:
            self.root_node = MCTSNode(board, player=self.player)
        else:
            #print("here")
            self.root_node = self.root_node.find_child_by_state(board)
        self.root_node.mcts(self.simulations_per_turn, network=self.model)

        #best_child = self.root_node.get_best_child()
        #move = best_child.move
        self.model.eval()

        Ns = np.zeros(9)
        nodes = np.empty(9, dtype=object)
        #nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
        for child in self.root_node.children:
            move = child.move
            # convert to index
            move_index = move[0] * 3 + move[1]
            Ns[move_index] = child.N
            nodes[move_index] = child

        probs = dist(Ns, tau=0.) # dist is a function that returns a probability distribution over the nodes
        self.probs_list.append(probs)
        next_node = np.random.choice(nodes, p=probs)
        move = next_node.move

        if self.verbose:
            # Get Ns and Qs for debugging or analysis purposes
            Ns_ = self.root_node.get_Ns()
            Qs_ = self.root_node.get_Qs()
            print(Ns_)
            print(np.sum(Ns_))
            print(Qs_)
            print("+"*45)
        
        self.root_node = next_node

        return move

def dist(x, tau=1):
    if tau == 0:
        probs = np.zeros_like(x)
        probs[np.argmax(x)] = 1
        return probs
    elif tau == np.inf:
        # return equal probabilities for all entries
        probs = np.ones_like(x) / np.count_nonzero(x)
        probs[x == 0] = 0
        return probs
    else:
        return np.power(x, 1 / tau) / np.sum(np.power(x, 1 / tau))



def initialize_players():
    """Initializes and returns two players for the game."""
    player1 = BetaZeroPlayer(1, model, verbose=False)
    player2 = BetaZeroPlayer(-1, model, verbose=False)
    return player1, player2

def play_game(player1, player2):
    """Plays a game and returns the game states, and result."""
    game_manager = TicTacToeHeadless(player1, player2)
    states, _, result = game_manager.play()
    return states, result

def process_probabilities(probs1, probs2):
    """Interleaves and processes two lists of probabilities."""
    probs = [torch.tensor(item) for pair in zip(probs1, probs2) for item in pair]
    probs.extend(torch.tensor(probs1[len(probs2):] if len(probs1) > len(probs2) else probs2[len(probs1):]))
    return probs

def prepare_training_data(states, probs, result):
    """Prepares and returns game data for training."""
    state_tensors = [convert_state_to_tensor(state, player_perpective=-(2*(i%2)-1)) for i, state in enumerate(states)]
    result_tensors = [torch.tensor([result * (-1)**i], dtype=torch.float32) for i in range(len(state_tensors))]
    state_tensors = torch.stack(state_tensors)
    probs_tensors = torch.stack(probs)
    result_tensors = torch.stack(result_tensors)
    return state_tensors, probs_tensors, result_tensors

def simulate_and_prepare_game_data():
    """Simulates a game and prepares the data for training."""
    player1, player2 = initialize_players()
    states, result = play_game(player1, player2)
    probs = process_probabilities(player1.probs_list, player2.probs_list)
    return prepare_training_data(states, probs, result)

def perform_simulations(num_games):
    all_states, all_probs, all_results = [], [], []

    for _ in range(num_games):
        states, probs, results = simulate_and_prepare_game_data()

        # Flatten and align each data point (state, probability, result)
        for state, prob, result in zip(states, probs, results):
            all_states.append(state)
            all_probs.append(prob)
            all_results.append(result)

    return all_states, all_probs, all_results


model = ConvolutionalTicTacToeNetWithDropout() #ResidualTicTacToeNet()

# Example Usage
num_games = 100 #500
all_states, all_probs, all_results = perform_simulations(num_games)


# Convert the lists to tensors
all_states_tensor = torch.stack(all_states)
all_probs_tensor = torch.stack(all_probs)
all_results_tensor = torch.stack(all_results)

print("Shapes:")
print(all_states_tensor.shape)
print(all_probs_tensor.shape)
print(all_results_tensor.shape)
out_probs, out_val = model(all_states_tensor.float())
print(out_probs.shape)
print(out_val.shape)
print()

# Create a TensorDataset
dataset = TensorDataset(all_states_tensor, all_probs_tensor, all_results_tensor)

# Now dataset can be used for model training
print("Dataset Length:", len(dataset))

# Assuming 'dataset' is your original full dataset
train_size = int(0.8 * len(dataset))  # 80% of data for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 256
val_batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

train_loss_values = []
val_loss_values = []

# AlphaGo Zero Loss
def alphago_zero_loss(policy_output, value_output, true_policy, true_value, c=1e-4):
    #print("-"*45)
    #print(value_output.shape)
    #print(true_value.shape)
    #print("-"*45)
    value_loss = nn.MSELoss()(value_output, true_value)
    policy_loss = -torch.mean(torch.sum(true_policy * torch.log(policy_output), 1))
    l2_reg = c * sum(torch.sum(param ** 2) for param in model.parameters())
    
    total_loss = value_loss + policy_loss + l2_reg
    return total_loss

optimizer = optim.Adam(model.parameters(), 0.001)#lr=0.0001)

# Set up live plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Initialize plot lines
line1, = ax.plot(train_loss_values, label='Training Loss')
line2, = ax.plot(val_loss_values, label='Validation Loss', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

num_epochs = 250
for epoch in tqdm(range(num_epochs), desc='Training Progress'):
    # Training Phase
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for states, probs, results in train_loader:
        policy_output, value_output = model(states.float())
        loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches
    train_loss_values.append(avg_train_loss)

    # Validation Phase
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    with torch.no_grad():
        for states, probs, results in val_loader:
            policy_output, value_output = model(states.float())
            loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())
            total_val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    val_loss_values.append(avg_val_loss)

    # Update plot
    line1.set_xdata(range(len(train_loss_values)))
    line1.set_ydata(train_loss_values)
    if epoch == 0:  # Add validation line after first epoch
        line2, = ax.plot(val_loss_values, label='Validation Loss')
        plt.legend()
    else:
        line2.set_xdata(range(len(val_loss_values)))
        line2.set_ydata(val_loss_values)
    ax.relim()  
    ax.autoscale_view(True, True, True)  
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()

## Try to understand if it's learning a viable v function and probs

# Create a board instance
board = TicTacToeBoard()
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((0, 2), 1)
# board = board.make_move((2, 0), -1)
# board = board.make_move((1, 2), 1)
# board = board.make_move((1, 1), -1)

board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((1, 1), 1)
board = board.make_move((0, 2), -1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perpective=1)
P, v = model(s.unsqueeze(0).float())
print(v)
print(P.squeeze().view(3, 3))

root_node = MCTSNode(board, player=1)
root_node.mcts(1000, network=model)

Ns = np.zeros(9)
nodes = np.empty(9, dtype=object)
#nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
for child in root_node.children:
    move = child.move
    # convert to index
    move_index = move[0] * 3 + move[1]
    Ns[move_index] = child.N
    nodes[move_index] = child

probs = dist(Ns, tau=1.)
print(probs.reshape(3, 3))
print(root_node.get_Ns().reshape(3, 3))
print(root_node.get_Qs().reshape(3, 3))
print(-root_node.Q)


#####

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((0, 2), 1)
# board = board.make_move((2, 0), -1)
# board = board.make_move((1, 2), 1)
# board = board.make_move((1, 1), -1)

board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((1, 1), 1)
board = board.make_move((2, 0), -1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perpective=1)
P, v = model(s.unsqueeze(0).float())
print(v)
print(P.squeeze().view(3, 3))

root_node = MCTSNode(board, player=1)
root_node.mcts(1000, network=model)

Ns = np.zeros(9)
nodes = np.empty(9, dtype=object)
#nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
for child in root_node.children:
    move = child.move
    # convert to index
    move_index = move[0] * 3 + move[1]
    Ns[move_index] = child.N
    nodes[move_index] = child

probs = dist(Ns, tau=1.)
print(probs.reshape(3, 3))
print(root_node.get_Ns().reshape(3, 3))
print(root_node.get_Qs().reshape(3, 3))
print(-root_node.Q)


#####

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((0, 2), 1)
# board = board.make_move((2, 0), -1)
# board = board.make_move((1, 2), 1)
# board = board.make_move((1, 1), -1)

board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((0, 2), 1)
board = board.make_move((1, 1), -1)
board = board.make_move((2, 0), 1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perpective=-1)
P, v = model(s.unsqueeze(0).float())
print(v)
print(P.squeeze().view(3, 3))

root_node = MCTSNode(board, player=-1)
root_node.mcts(1000, network=model)

Ns = np.zeros(9)
nodes = np.empty(9, dtype=object)
#nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
for child in root_node.children:
    move = child.move
    # convert to index
    move_index = move[0] * 3 + move[1]
    Ns[move_index] = child.N
    nodes[move_index] = child

probs = dist(Ns, tau=1.)
print(probs.reshape(3, 3))
print(root_node.get_Ns().reshape(3, 3))
print(root_node.get_Qs().reshape(3, 3))
print(-root_node.Q)


####

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((0, 2), 1)
# board = board.make_move((2, 0), -1)
# board = board.make_move((1, 2), 1)
# board = board.make_move((1, 1), -1)

board = board.make_move((0, 1), 1)
board = board.make_move((1, 1), -1)
board = board.make_move((0, 2), 1)
board = board.make_move((0, 0), -1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perpective=1)
P, v = model(s.unsqueeze(0).float())
print(v)
print(P.squeeze().view(3, 3))

root_node = MCTSNode(board, player=1)
root_node.mcts(1000, network=model)

Ns = np.zeros(9)
nodes = np.empty(9, dtype=object)
#nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
for child in root_node.children:
    move = child.move
    # convert to index
    move_index = move[0] * 3 + move[1]
    Ns[move_index] = child.N
    nodes[move_index] = child

probs = dist(Ns, tau=1.)
print(probs.reshape(3, 3))
print(root_node.get_Ns().reshape(3, 3))
print(root_node.get_Qs().reshape(3, 3))
print(-root_node.Q)