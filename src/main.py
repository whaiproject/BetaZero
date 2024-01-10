import torch.optim as optim
from network import ConvolutionalTicTacToeNetWithDropout
from trainer import AgentTrainer

# only for testing
from utils import convert_state_to_tensor, dist
from mcts import MCTSNode
import numpy as np
from ttt.tic_tac_toe_board import TicTacToeBoard

model = ConvolutionalTicTacToeNetWithDropout()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = AgentTrainer(model, optimizer, num_iterations=5, num_games_per_iteration=100, num_epochs_per_iteration=100, batch_size=256)
trainer.train()


## test

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
s = convert_state_to_tensor(board.board, player_perspective=1)
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
s = convert_state_to_tensor(board.board, player_perspective=1)
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


##### Testing

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()

board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((0, 2), 1)
board = board.make_move((1, 1), -1)
board = board.make_move((2, 0), 1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perspective=-1)
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
s = convert_state_to_tensor(board.board, player_perspective=1)
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


####

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()
board = board.make_move((0, 0), 1)
board = board.make_move((1, 1), -1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perspective=1)
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

####

print("~"*45)
print("~"*45)

# Create a board instance
board = TicTacToeBoard()

board = board.make_move((0, 0), 1)
board = board.make_move((0, 1), -1)
board = board.make_move((0, 2), 1)
board = board.make_move((1, 1), -1)
board = board.make_move((1, 0), 1)

print(board)

model.eval()
s = convert_state_to_tensor(board.board, player_perspective=-1)
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

# It's very bad at predicting v and probs from the perspective of player -1
# it's decent for player 1

# If I run it with more than 2 iterations with the evaulate function that
# uses the network, it returns nan for v and probs. WHY??