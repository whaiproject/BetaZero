import numpy as np
import torch
import torch.nn.functional as F

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, player=None, prob=1.):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.N = 0  # Visit count
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.c = 0.5  # Exploration constant
        self.P = prob  # Prior probability
        self.depth = 0 if parent is None else parent.depth + 1
        self.player = player if player else -self.parent.player

    def select(self):
        """ Select the leaf node using UCT algorithm. """
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
        return P.squeeze().detach().numpy(), v.item()

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

    def mcts(self, iterations, network):
        """ Perform MCTS for a given number of iterations. """
        for _ in range(iterations):
            node = self.select()
            if not node.game_state.is_game_over():
                probs, _ = node.evaluate(network)  # Get probabilities
                node.expand(probs)

            result = node.evaluate(network)[1]  # Get value
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
 


def convert_state_to_tensor(s_raw, player_perpective=1):
    s = torch.zeros((2, 3, 3), dtype=torch.int32)
    s[0, s_raw == player_perpective] = 1
    s[1, s_raw == -player_perpective] = 1
    return s



class BetaZeroPlayer(Player):
    def __init__(self, player, model, verbose=False):
        super().__init__()
        self.player = player
        self.root_node = None
        self.simulations_per_turn = 25 #25
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


