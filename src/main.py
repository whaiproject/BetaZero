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
            #print(current_state.board)
            current_player *= -1

        self.P = 1 # change this to be set to the output of the neural net
        return current_state.get_game_result()

    def backpropagate(self, result):
        current_node = self
        #print("??"*20)
        while current_node is not None:
            # print("+"*20)
            # print(result)
            # print(current_node.player)
            # print(result * current_node.player)
            # print("+"*20)
            current_node.N += 1
            # the - here is needed because the result is from the perspective of the
            # player who just moved which is the parent of the current node
            current_node.W += -result * current_node.player
            current_node.Q = current_node.W / current_node.N
            current_node = current_node.parent


def mcts(root_state, iterations, player=1):
    root_node = MCTSNode(root_state, player=player)

    # perform various simulations
    for _ in range(iterations):
        node = root_node.select()
        if not node.game_state.is_game_over():
            node.expand()

        # continue with lightweight policy
        result = node.evaluate()
        node.backpropagate(result)

    return root_node, max(root_node.children, key=lambda node: node.N).move



# class RandomPlayer(Player):
#     def get_move(self, board):
#         return random.choice(board.get_valid_moves())
    

class BetaZeroPlayer(Player):
    def __init__(self, player):
        super().__init__()
        self.player = player

    def get_move(self, board):
        root_node, move = mcts(board, 1000, player=self.player)
        Ns = np.zeros((3, 3))
        Qs = np.zeros((3, 3))
        for node in root_node.children:
            Ns[node.move] = node.N
            Qs[node.move] = node.Q
        print(Ns)
        print(Qs)
        print("#"*20)
        return move


player1 = BetaZeroPlayer(1)
player2 = RandomPlayer()

# Initialize the terminal game manager
terminal_game_manager = TicTacToeTerminal(player1, player2)
terminal_game_manager.play()


# board = TicTacToeBoard()
# board = board.make_move((0, 0), 1)
# board = board.make_move((0, 1), -1)
# board = board.make_move((1, 1), 1)
# board = board.make_move((2, 1), -1)


# print(board.board)

# root_node, a = mcts(board, 100, player=1)

# #Ns = np.array([node.N for node in root_node.children]).reshape(3, 3)
# Ns = np.zeros((3, 3))
# for node in root_node.children:
#     Ns[node.move] = node.N

# print(root_node.N)
# print(Ns)
# print(a)