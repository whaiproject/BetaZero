from ttt.tic_tac_toe_board import TicTacToeBoard
from ttt.tic_tac_toe import TicTacToeTerminal
from ttt.players import Player, RandomPlayer, OptimalPlayer
import numpy as np
import random

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
                key=lambda node: node.Q + self.c * np.sqrt(node.parent.N) / (1 + node.N)
            )
        return current_node

    def expand(self):
        valid_moves = self.game_state.get_valid_moves()
        for move in valid_moves:
            next_state = self.game_state.make_move(move, self.player)
            child_node = MCTSNode(next_state, parent=self, move=move)
            self.children.append(child_node)

    # Evaluate position (through random play for example (like rollout policy))
    def evaluate(self):
        current_state = self.game_state

        current_player = self.player
        # change this to set v to the output of the neural net
        while not current_state.is_game_over():
            possible_moves = current_state.get_valid_moves()
            move = random.choice(possible_moves)
            current_state = current_state.make_move(move, current_player)
            current_player *= -1

        self.P = 1 # change this to be set to the output of the neural net
        return current_state.get_game_result()

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

    def mcts(self, iterations):

        # perform various simulations
        for _ in range(iterations):
            node = self.select()
            if not node.game_state.is_game_over():
                node.expand()

            # continue with lightweight policy
            result = node.evaluate()
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
 

class BetaZeroPlayer(Player):
    def __init__(self, player, verbose=False):
        super().__init__()
        self.player = player
        self.root_node = None
        self.simulations_per_turn = 1000
        self.verbose = verbose

    def get_move(self, board):
        if self.root_node is None:
            self.root_node = MCTSNode(board, player=self.player)
        else:
            self.root_node = self.root_node.find_child_by_state(board)
        self.root_node.mcts(1000)

        best_child = self.root_node.get_best_child()
        move = best_child.move

        if self.verbose:
            # Get Ns and Qs for debugging or analysis purposes
            Ns = self.root_node.get_Ns()
            Qs = self.root_node.get_Qs()
            print(Ns)
            print(Qs)
            print("+"*45)
        
        self.root_node = best_child

        return move


player1 = RandomPlayer()
player2 = BetaZeroPlayer(-1, verbose=True)

# Initialize the terminal game manager
terminal_game_manager = TicTacToeTerminal(player1, player2)
terminal_game_manager.play()
