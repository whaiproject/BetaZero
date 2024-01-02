from ttt.tic_tac_toe_board import TicTacToeBoard
from ttt.tic_tac_toe import TicTacToeTerminal, TicTacToeHeadless
from ttt.players import Player, RandomPlayer, OptimalPlayer
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, player=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.N = 0
        self.W = 0
        self.Q = 0
        self.c = .5
        self.P = None
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
                key=lambda node: node.Q + self.c * np.sqrt(node.parent.N) / (1 + node.N) # add self.P when implemented
            )
        return current_node

    def expand(self):
        valid_moves = self.game_state.get_valid_moves()
        for move in valid_moves:
            next_state = self.game_state.make_move(move, self.player)
            child_node = MCTSNode(next_state, parent=self, move=move)
            self.children.append(child_node)

    # Evaluate position (through random play for example (like rollout policy))
    def evaluate(self, network):
        # s = convert_state_to_tensor(self.game_state.board, player_perpective=self.player)
        # P, v = network(s.unsqueeze(0).float())

        current_state = self.game_state
        current_player = self.player
        # change this to set v to the output of the neural net
        while not current_state.is_game_over():
            possible_moves = current_state.get_valid_moves()
            move = random.choice(possible_moves)
            current_state = current_state.make_move(move, current_player)
            current_player *= -1

        self.P = P # change this to be set to the output of the neural net
        return current_state.get_game_result() # v

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

        # perform various simulations
        for _ in range(iterations):
            node = self.select()
            if not node.game_state.is_game_over():
                node.expand()

            # continue with lightweight policy
            result = node.evaluate(network=network)
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

# Create an instance of the network
net = TicTacToeNet()

board = TicTacToeBoard()
# Modified to pass tuples
board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((0, 2), 1)
s_raw = torch.tensor(board.board)

def convert_state_to_tensor(s_raw, player_perpective=1):
    s = torch.zeros((2, 3, 3), dtype=torch.int32)
    s[0, s_raw == player_perpective] = 1
    s[1, s_raw == -player_perpective] = 1
    return s

# Usage
s = convert_state_to_tensor(s_raw, player_perpective=-1)
print(s_raw)
print(s)

P, v = net(s.unsqueeze(0).float())
print(P)
print(v)

class BetaZeroPlayer(Player):
    def __init__(self, player, verbose=False):
        super().__init__()
        self.player = player
        self.root_node = None
        self.simulations_per_turn = 100
        self.verbose = verbose
        self.probs_list = []

    def get_move(self, board):
        if self.root_node is None:
            self.root_node = MCTSNode(board, player=self.player)
        else:
            self.root_node = self.root_node.find_child_by_state(board)
        self.root_node.mcts(self.simulations_per_turn, network=net)

        #best_child = self.root_node.get_best_child()
        #move = best_child.move

        Ns = np.zeros(9)
        nodes = np.empty(9, dtype=object)
        #nodes, Ns = zip(*[(child, child.N) for child in self.root_node.children])
        for child in self.root_node.children:
            move = child.move
            # convert to index
            move_index = move[0] * 3 + move[1]
            Ns[move_index] = child.N
            nodes[move_index] = child

        probs = dist(Ns) # dist is a function that returns a probability distribution over the nodes
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
    return np.power(x, 1 / tau) / np.sum(np.power(x, 1 / tau))

player1 = BetaZeroPlayer( 1, verbose=False)
player2 = BetaZeroPlayer(-1, verbose=False)

# Initialize the terminal game manager
terminal_game_manager = TicTacToeHeadless(player1, player2)
states, _, result = terminal_game_manager.play()
probs1 = player1.probs_list
probs2 = player2.probs_list
probs = [torch.tensor(item) for pair in zip(probs1, probs2) for item in pair]
# Extend with the remaining elements from the longer list
if len(probs1) > len(probs2):
    probs.extend(torch.tensor(probs1[len(probs2):]))
else:
    probs.extend(torch.tensor(probs2[len(probs1):]))

game_data = [states, probs, result]

print("Probs: ")
print(probs)

def prepare_training_data(game_data):
    states, probs, result = game_data[0], game_data[1], game_data[2]

    # Convert states to tensor format
    # add alternating player perspective
    state_tensors = [convert_state_to_tensor(state, player_perpective=-(2*(i%2)-1)) for i, state in enumerate(states)]

    # # One-hot encode the moves
    # move_labels = [torch.zeros(9, dtype=torch.float32) for _ in moves]
    # for label, move in zip(move_labels, moves):
    #     #print(move)
    #     row, col = move
    #     index = row * 3 + col
    #     label[index] = 1.0

    # Convert results to tensor format
    #result_tensors = [torch.tensor([result], dtype=torch.float32) for result in results]
    #result_tensors = [result * player for result, player in zip(result_tensors, [1, -1] * (len(state_tensors) // 2))]
    result_tensors = [torch.tensor([result * (-1)**i], dtype=torch.float32) for i in range(len(state_tensors))]

    # Stack all tensors
    state_tensors = torch.stack(state_tensors)
    probs_tensors = torch.stack(probs)
    result_tensors = torch.stack(result_tensors)

    return state_tensors, probs_tensors, result_tensors

print("x"*45)
# Example usage
res = prepare_training_data(game_data)
print(res)

print(game_data[2])

# player1 = RandomPlayer()
# player2 = BetaZeroPlayer(-1, verbose=False)
# # Initialize the terminal game manager
# terminal_game_manager = TicTacToeTerminal(player1, player2)
# game_data = terminal_game_manager.play()