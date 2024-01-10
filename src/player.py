import numpy as np
from ttt.players import Player
from mcts import MCTSNode
from utils import dist

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
        # Initialize or update the root node
        self.root_node = MCTSNode(board, player=self.player) if self.root_node is None else self.root_node.find_child_by_state(board)
        
        # Run MCTS
        self.root_node.mcts(self.simulations_per_turn, network=self.model)

        # Set model to evaluation mode
        self.model.eval()

        # Prepare arrays for storing move probabilities and nodes
        Ns = np.zeros(9)
        nodes = np.empty(9, dtype=object)

        # Populate the arrays with data from child nodes
        for child in self.root_node.children:
            move_index = child.move[0] * 3 + child.move[1]
            Ns[move_index] = child.N
            nodes[move_index] = child

        # Get move probabilities and choose the next node
        probs = dist(Ns, tau=0.)  # Assume dist is a predefined function
        self.probs_list.append(probs)
        next_node = np.random.choice(nodes, p=probs)
        move = next_node.move

        # Verbose mode for debugging or analysis
        if self.verbose:
            Ns_debug = self.root_node.get_Ns()
            Qs_debug = self.root_node.get_Qs()
            print(Ns_debug)
            print(np.sum(Ns_debug))
            print(Qs_debug)
            print("+" * 45)

        # Update the root node for the next turn
        self.root_node = next_node

        return move
