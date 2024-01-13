import numpy as np
import torch.nn.functional as F
from utils import convert_state_to_tensor

# for testing
import random

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, player=None, prob=1.):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.N = 0  # Visit count
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.c = 0.3 #0.3  # Exploration constant
        self.P = prob  # Prior probability
        self.depth = 0 if parent is None else parent.depth + 1
        self.player = player if player else -self.parent.player

    def select(self):
        """ Select the leaf node using a modified version of the PUCT algorithm. """
        current_node = self
        while current_node.children:
            current_node = max(
                current_node.children, 
                key=lambda node: node.Q + self.c * self.P * np.sqrt(node.parent.N) / (1 + node.N)
            )
        return current_node

    def expand(self, probs):
        """ Expand the node by adding all valid child nodes. """
        valid_moves = self.game_state.get_valid_moves()
        for i, move in enumerate(valid_moves):
            next_state = self.game_state.make_move(move, self.player)
            child_node = MCTSNode(next_state, parent=self, move=move, prob=probs[i])
            self.children.append(child_node)

    def evaluate(self, network):
        """ Evaluate the game state using the given network. """
        tensor_state = convert_state_to_tensor(self.game_state.board, player_perspective=self.player)
        P, v = network(tensor_state.unsqueeze(0).float())
        return P.squeeze().detach().numpy(), v.item() * self.player

    def evaluate2(self, network):
        s = convert_state_to_tensor(self.game_state.board, player_perspective=self.player)
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

    def backpropagate(self, result):
        """ Update the node and its ancestors with the simulation result. """
        current_node = self
        while current_node:
            current_node.N += 1
            current_node.W += -result * current_node.player  # Invert result for parent's perspective
            current_node.Q = current_node.W / current_node.N
            current_node = current_node.parent

    def find_child_by_move(self, move):
        """ Find a child node corresponding to the given move. """
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
        """ Perform MCTS for a given number of iterations. """
        for _ in range(iterations):
            node = self.select()
            if not node.game_state.is_game_over():
                probs, _ = node.evaluate(network)  # Get probabilities
                node.expand(probs)

            # shouldn't we use the actual result herer?
            # the network is never trained on terminated positions
            # so it will never learn to predict the result
            # of an already ended game
            # I was effectively using trash evaluations at many
            # leaf nodes of the MCTS
            
            if node.game_state.is_game_over():
                result = node.game_state.get_game_result()
            else:
                result = node.evaluate(network)[1]  # Get value # I am also doing the forward pass twice which is inefficient
            node.backpropagate(result)

    def get_best_child(self):
        """ Get the child with the highest visit count. """
        return max(self.children, key=lambda node: node.N)

    def get_Ns(self):
        """ Get an array of visit counts for each move. """
        Ns = np.zeros((3, 3))
        for node in self.children:
            Ns[node.move] = node.N
        return Ns

    def get_Qs(self):
        """ Get an array of mean values for each move. """
        Qs = np.zeros((3, 3))
        for node in self.children:
            Qs[node.move] = node.Q
        return Qs

