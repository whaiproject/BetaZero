import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from player import BetaZeroPlayer
from ttt.players import RandomPlayer
from ttt.tic_tac_toe import TicTacToeHeadless
from utils import convert_state_to_tensor, alphago_zero_loss
import collections
import matplotlib.pyplot as plt


class AgentTrainer:
    def __init__(self, model, optimizer, num_iterations, num_games_per_iteration, num_epochs_per_iteration, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.game_data_queue = collections.deque(maxlen=500)  # Queue to store game data

    def train(self):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))

        p1_list, p2_list, draw_list = [], [], []
        iterations = []

        for iteration in range(self.num_iterations):

            if iteration % 5 == 0:
                iterations.append(iteration)
                # Test against random player
                num_test_games = 1000
                p1, p2, draw = 0, 0, 0
                for _ in range(num_test_games):
                    player1, player2 = self.initialize_test_players_random()
                    _, result = self.play_game(player1, player2)
                    if result == 1:
                        p1 += 1
                    elif result == -1:
                        p2 += 1
                    else:
                        draw += 1.

                p1 = p1 / num_test_games * 100
                p2 = p2 / num_test_games * 100
                draw = draw / num_test_games * 100

                print(f"Iteration {iteration}: p1: {p1}%, p2: {p2}%, Draw: {draw}%")

                p1_list.append(p1)
                p2_list.append(p2)
                draw_list.append(draw)

                ax.clear()
                ax.plot(iterations, p1_list, label='p1 Percentage')
                ax.plot(iterations, p2_list, label='p2 Percentage')
                ax.plot(iterations, draw_list, label='Draw Percentage')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Percentage')
                ax.set_title('Performance Over Time')
                ax.legend()
                plt.draw()
                plt.pause(0.001)

            # Perform training
            self.simulate_games()
            self.train_model()
            print(f"Iteration {iteration + 1}/{self.num_iterations} completed.")

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot


    def plot_results(self, iterations, wins, losses, draws):
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, wins, label='Win Percentage')
        plt.plot(iterations, losses, label='Loss Percentage')
        plt.plot(iterations, draws, label='Draw Percentage')
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        plt.title('Performance Over Time')
        plt.legend()
        plt.show()

    def initialize_players(self):
        player1 = BetaZeroPlayer(1, self.model, verbose=False, tau=0.5)
        player2 = BetaZeroPlayer(-1, self.model, verbose=False, tau=0.5)
        return player1, player2
    
    def initialize_test_players_random(self):
        # player1 = BetaZeroPlayer(1, self.model, verbose=False)
        # player2 = RandomPlayer()
        player1 = RandomPlayer()
        player2 = BetaZeroPlayer(-1, self.model, verbose=False)
        return player1, player2

    def play_game(self, player1, player2):
        game_manager = TicTacToeHeadless(player1, player2)
        states, _, result = game_manager.play()
        return states, result

    def process_probabilities(self, probs1, probs2):
        probs = [torch.tensor(item) for pair in zip(probs1, probs2) for item in pair]
        probs.extend(torch.tensor(probs1[len(probs2):] if len(probs1) > len(probs2) else probs2[len(probs1):]))
        return probs

    def prepare_training_data(self, states, probs, result):
        state_tensors = [convert_state_to_tensor(state, player_perspective=-(2*(i%2)-1)) for i, state in enumerate(states)]
        result_tensors = [torch.tensor([result * (-1)**i], dtype=torch.float32) for i in range(len(state_tensors))]
        return torch.stack(state_tensors), torch.stack(probs), torch.stack(result_tensors)

    def simulate_and_prepare_game_data(self):
        player1, player2 = self.initialize_players()
        states, result = self.play_game(player1, player2)
        probs = self.process_probabilities(player1.probs_list, player2.probs_list)
        return self.prepare_training_data(states, probs, result)

    def simulate_games(self):
        for _ in range(self.num_games_per_iteration):
            states, probs, results = self.simulate_and_prepare_game_data()
            game_data = (states, probs, results)
            self.game_data_queue.append(game_data)  # Add new game data to the queue

    def train_model(self):
        all_states, all_probs, all_results = [], [], []
        for states, probs, results in self.game_data_queue:  # Use data from the queue
            all_states.extend(states)
            all_probs.extend(probs)
            all_results.extend(results)
        
        dataset = self.prepare_dataset(all_states, all_probs, all_results)
        train_loader, val_loader = self.prepare_data_loaders(dataset)
        for epoch in range(self.num_epochs_per_iteration):
            self.run_training_epoch(train_loader)
            self.run_validation_epoch(val_loader)

    def prepare_dataset(self, all_states, all_probs, all_results):
        all_states_tensor = torch.stack(all_states)
        all_probs_tensor = torch.stack(all_probs)
        all_results_tensor = torch.stack(all_results)
        return TensorDataset(all_states_tensor, all_probs_tensor, all_results_tensor)

    def prepare_data_loaders(self, dataset):
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # print("@"*45)
        # print("@"*45)
        # print(train_dataset[:])
        # print("@"*45)
        # print("@"*45)
        return train_loader, val_loader

    def run_training_epoch(self, train_loader):
        self.model.train()
        for states, probs, results in train_loader:
            policy_output, value_output = self.model(states.float())
            loss = alphago_zero_loss(self.model, policy_output, value_output, probs.float(), results.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_validation_epoch(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            for states, probs, results in val_loader:
                policy_output, value_output = self.model(states.float())
                alphago_zero_loss(self.model, policy_output, value_output, probs.float(), results.float())







########################################################################################################################
########################################################################################################################
########################################################################################################################









class AgentTrainer2:
    def __init__(self, model, optimizer, num_iterations, num_games_per_iteration, num_epochs_per_iteration, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size

    def train(self):
        for iteration in range(self.num_iterations):

            if iteration % 5 == 0:    
                # test against random player
                num_test_games = 100
                p1, p2, draw = 0, 0, 0
                for _ in range(num_test_games):
                    player1, player2 = self.initialize_test_players_random()
                    _, result = self.play_game(player1, player2)
                    if result == 1:
                        p1 += 1
                    elif result == -1:
                        p2 += 1
                    else:
                        draw += 1
                print(f"p1: {p1/num_test_games * 100}%, p2: {p2/num_test_games * 100}%, draw: {draw/num_test_games * 100}%")

            # perform training
            all_states, all_probs, all_results = self.simulate_games()
            self.train_model(all_states, all_probs, all_results)
            print(f"Iteration {iteration + 1}/{self.num_iterations} completed.")

    def initialize_players(self):
        player1 = BetaZeroPlayer(1, self.model, verbose=False)
        player2 = BetaZeroPlayer(-1, self.model, verbose=False)
        return player1, player2
    
    def initialize_test_players_random(self):
        player1 = RandomPlayer()
        player2 = BetaZeroPlayer(-1, self.model, verbose=False)
        return player1, player2

    def play_game(self, player1, player2):
        game_manager = TicTacToeHeadless(player1, player2)
        states, _, result = game_manager.play()
        return states, result

    def process_probabilities(self, probs1, probs2):
        probs = [torch.tensor(item) for pair in zip(probs1, probs2) for item in pair]
        probs.extend(torch.tensor(probs1[len(probs2):] if len(probs1) > len(probs2) else probs2[len(probs1):]))
        return probs

    def prepare_training_data(self, states, probs, result):
        state_tensors = [convert_state_to_tensor(state, player_perspective=-(2*(i%2)-1)) for i, state in enumerate(states)]
        result_tensors = [torch.tensor([result * (-1)**i], dtype=torch.float32) for i in range(len(state_tensors))]
        return torch.stack(state_tensors), torch.stack(probs), torch.stack(result_tensors)

    def simulate_and_prepare_game_data(self):
        player1, player2 = self.initialize_players()
        states, result = self.play_game(player1, player2)
        probs = self.process_probabilities(player1.probs_list, player2.probs_list)
        return self.prepare_training_data(states, probs, result)

    def simulate_games(self):
        all_states, all_probs, all_results = [], [], []
        for _ in range(self.num_games_per_iteration):
            states, probs, results = self.simulate_and_prepare_game_data()
            all_states.extend(states)
            all_probs.extend(probs)
            all_results.extend(results)
        return all_states, all_probs, all_results

    def train_model(self, all_states, all_probs, all_results):
        dataset = self.prepare_dataset(all_states, all_probs, all_results)
        train_loader, val_loader = self.prepare_data_loaders(dataset)
        for epoch in range(self.num_epochs_per_iteration):
            self.run_training_epoch(train_loader)
            self.run_validation_epoch(val_loader)

    def prepare_dataset(self, all_states, all_probs, all_results):
        all_states_tensor = torch.stack(all_states)
        all_probs_tensor = torch.stack(all_probs)
        all_results_tensor = torch.stack(all_results)
        return TensorDataset(all_states_tensor, all_probs_tensor, all_results_tensor)

    def prepare_data_loaders(self, dataset):
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        print("@"*45)
        print("@"*45)
        print(train_dataset[:])
        print("@"*45)
        print("@"*45)
        return train_loader, val_loader

    def run_training_epoch(self, train_loader):
        self.model.train()
        for states, probs, results in train_loader:
            policy_output, value_output = self.model(states.float())
            loss = alphago_zero_loss(self.model, policy_output, value_output, probs.float(), results.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_validation_epoch(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            for states, probs, results in val_loader:
                policy_output, value_output = self.model(states.float())
                alphago_zero_loss(self.model, policy_output, value_output, probs.float(), results.float())
