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
        return P.squeeze().detach().numpy(), v.item()

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
        self.simulations_per_turn = 15
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

        probs = dist(Ns, tau=0.3) # dist is a function that returns a probability distribution over the nodes
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

def alphago_zero_loss(policy_output, value_output, true_policy, true_value, c=1e-4):
    value_loss = nn.MSELoss()(value_output.squeeze(), true_value)
    policy_loss = -torch.mean(torch.sum(true_policy * torch.log(policy_output), 1))
    l2_reg = c * sum(torch.sum(param ** 2) for param in model.parameters())
    
    total_loss = value_loss + policy_loss + l2_reg
    return total_loss

model = TicTacToeNet()  # Create an instance of the model

def main_loop(num_iterations, num_games_per_iteration, num_epochs_per_training, num_test_games):

    p1_wins = []
    p2_wins = []
    game_draws = []

    # Testing phase
    num_games = 200  # Number of test games
    p1, p2, draws = test_player(num_games)
    print(f"Wins: {p1}, Losses: {p2}, Draws: {draws}")

    p1_wins.append(p1)
    p2_wins.append(p2)
    game_draws.append(draws)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 4))


    for iteration in range(num_iterations):
        print(f"Starting Iteration {iteration + 1}/{num_iterations}")

        # Step 1: Perform simulations to generate training data
        all_states, all_probs, all_results = perform_simulations(num_games_per_iteration)

        # Step 2: Convert the lists to tensors
        all_states_tensor = torch.stack(all_states)
        all_probs_tensor = torch.stack(all_probs)
        all_results_tensor = torch.stack(all_results)

        # Step 3: Create a TensorDataset
        dataset = TensorDataset(all_states_tensor, all_probs_tensor, all_results_tensor)

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

        # Step 4: Train the network
        train_network(model, train_loader, val_loader, num_epochs_per_training)

        # Testing phase
        num_games = 200  # Number of test games
        p1, p2, draws = test_player(num_games)

        p1_wins.append(p1)
        p2_wins.append(p2)
        game_draws.append(draws)

        print(f"Player 1: {p1}, Player 2: {p2}, Draws: {draws}")

        # Update the plot
        ax.clear()  # Clear previous plot
        ax.plot(p1_wins, label='Player 1 Wins')
        ax.plot(p2_wins, label='Player 2 Wins')
        ax.plot(game_draws, label='Draws')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Games')
        ax.set_title('Game Outcomes by Iteration')
        ax.legend()
        plt.pause(0.001)  # Pause to update the plot


        # Optional: Save the model after each iteration
        torch.save(model.state_dict(), f'model_iteration_{iteration}.pth')

def test_player(num_games):
    p1 = 0
    p2 = 0
    draws = 0
    for _ in range(num_games):
        player1 = RandomPlayer()  # Replace with BetaZeroPlayer(-1, model, verbose=False) if needed
        player2 = BetaZeroPlayer(-1, model, verbose=False)
        game_manager = TicTacToeHeadless(player1, player2)
        _, _, result = game_manager.play()
        if result == 1:
            p1 += 1
        elif result == -1:
            p2 += 1
        else:  # result == 0
            draws += 1
    return p1, p2, draws

def train_network(model, train_loader, val_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(num_epochs), desc='Epochs', leave=False):
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

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches else 0

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

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches else 0

        #print(f"\rEpoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}", end='')
        #tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Usage
num_iterations = 100  # Number of times to repeat the simulation-training cycle
num_games_per_iteration = 10  # Number of games to simulate in each iteration
num_epochs_per_training = 5  # Number of epochs for training in each iteration
num_test_games = 40

main_loop(num_iterations, num_games_per_iteration, num_epochs_per_training, num_test_games)








# # Example Usage
# num_games = 50
# all_states, all_probs, all_results = perform_simulations(num_games)


# # Convert the lists to tensors
# all_states_tensor = torch.stack(all_states)
# all_probs_tensor = torch.stack(all_probs)
# all_results_tensor = torch.stack(all_results)

# # Create a TensorDataset
# dataset = TensorDataset(all_states_tensor, all_probs_tensor, all_results_tensor)

# # Now dataset can be used for model training
# print("Dataset Length:", len(dataset))

# # Assuming 'dataset' is your original full dataset
# train_size = int(0.8 * len(dataset))  # 80% of data for training
# val_size = len(dataset) - train_size  # 20% for validation
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# batch_size = 16
# val_batch_size = 100
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

# train_loss_values = []
# val_loss_values = []

# # AlphaGo Zero Loss
# def alphago_zero_loss(policy_output, value_output, true_policy, true_value, c=1e-4):
#     value_loss = nn.MSELoss()(value_output.squeeze(), true_value)
#     policy_loss = -torch.mean(torch.sum(true_policy * torch.log(policy_output), 1))
#     l2_reg = c * sum(torch.sum(param ** 2) for param in model.parameters())
    
#     total_loss = value_loss + policy_loss + l2_reg
#     return total_loss

# model = TicTacToeNet()  # Create an instance of the model
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Set up live plot
# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots()

# # Initialize plot lines
# line1, = ax.plot(train_loss_values, label='Training Loss')
# line2, = ax.plot(val_loss_values, label='Validation Loss', linestyle='--')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()

# num_epochs = 500
# for epoch in tqdm(range(num_epochs), desc='Training Progress'):
#     # Training Phase
#     model.train()
#     total_train_loss = 0
#     num_train_batches = 0

#     for states, probs, results in train_loader:
#         policy_output, value_output = model(states.float())
#         loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_train_loss += loss.item()
#         num_train_batches += 1

#     avg_train_loss = total_train_loss / num_train_batches
#     train_loss_values.append(avg_train_loss)

#     # Validation Phase
#     model.eval()
#     total_val_loss = 0
#     num_val_batches = 0
#     with torch.no_grad():
#         for states, probs, results in val_loader:
#             policy_output, value_output = model(states.float())
#             loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())
#             total_val_loss += loss.item()
#             num_val_batches += 1

#     avg_val_loss = total_val_loss / num_val_batches
#     val_loss_values.append(avg_val_loss)

#     # Update plot
#     line1.set_xdata(range(len(train_loss_values)))
#     line1.set_ydata(train_loss_values)
#     if epoch == 0:  # Add validation line after first epoch
#         line2, = ax.plot(val_loss_values, label='Validation Loss')
#         plt.legend()
#     else:
#         line2.set_xdata(range(len(val_loss_values)))
#         line2.set_ydata(val_loss_values)
#     ax.relim()  
#     ax.autoscale_view(True, True, True)  
#     fig.canvas.draw()
#     fig.canvas.flush_events()

# plt.ioff()
# plt.show()









# # # Number of samples to display
# # num_samples_to_display = len(dataset)

# # # Display the first few samples from the dataset
# # for i in range(num_samples_to_display):
# #     state, prob, result = dataset[i]
# #     print(f"Sample {i+1}:")
# #     print("State:\n", state)
# #     print("Probability:\n", prob)
# #     print("Result:\n", result)
# #     print("-" * 30)

# # # Create a DataLoader instance
# # batch_size = 2  # You can adjust the batch size as needed
# # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # Number of batches to display
# # num_batches_to_display = 4

# # # Iterate over the DataLoader
# # for i, (states, probs, results) in enumerate(dataloader):
# #     print(f"Batch {i+1}:")
# #     print("States:\n", states)
# #     print("Probabilities:\n", probs)
# #     print("Results:\n", results)
# #     print("-" * 30)

# #     if i >= num_batches_to_display - 1:
# #         break


# # # AlphaGo Zero Loss
# # def alphago_zero_loss(policy_output, value_output, true_policy, true_value, c=1e-4):
# #     value_loss = nn.MSELoss()(value_output.squeeze(), true_value)
# #     policy_loss = -torch.mean(torch.sum(true_policy * torch.log(policy_output), 1))
# #     l2_reg = c * sum(torch.sum(param ** 2) for param in model.parameters())
    
# #     total_loss = value_loss + policy_loss + l2_reg
# #     return total_loss

# # model = TicTacToeNet()  # Create an instance of the model
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # import matplotlib.pyplot as plt
# # from tqdm import tqdm

# # # Set up live plot
# # plt.ion()  # Turn on interactive mode
# # fig, ax = plt.subplots()
# # loss_values = []
# # line1, = ax.plot(loss_values, label='Training Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.title('Training Loss Over Epochs')
# # plt.legend()

# # num_epochs = 500

# # for epoch in tqdm(range(num_epochs), desc='Training Progress'):
# #     total_loss = 0
# #     num_batches = 0

# #     for states, probs, results in dataloader:
# #         # Forward pass
# #         policy_output, value_output = model(states.float())
# #         loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())

# #         # Backward and optimize
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         num_batches += 1

# #     avg_loss = total_loss / num_batches
# #     loss_values.append(avg_loss)

# #     # Update plot
# #     line1.set_xdata(range(len(loss_values)))
# #     line1.set_ydata(loss_values)
# #     ax.relim()  # Recalculate limits
# #     ax.autoscale_view(True, True, True)  # Autoscale
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()

# # plt.ioff()  # Turn off interactive mode
# # plt.show()

# # # num_epochs = 500
# # # for epoch in range(num_epochs):
# # #     for states, probs, results in dataloader:
# # #         # Forward pass
# # #         policy_output, value_output = model(states.float())
# # #         loss = alphago_zero_loss(policy_output, value_output, probs.float(), results.float())

# # #         # Backward and optimize
# # #         optimizer.zero_grad()
# # #         loss.backward()
# # #         optimizer.step()

# # #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
