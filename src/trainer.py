import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from player import BetaZeroPlayer
from ttt.tic_tac_toe import TicTacToeHeadless
from utils import convert_state_to_tensor, alphago_zero_loss

class AgentTrainer:
    def __init__(self, model, optimizer, num_iterations, num_games_per_iteration, num_epochs_per_iteration, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size

    def train(self):
        for iteration in range(self.num_iterations):
            all_states, all_probs, all_results = self.simulate_games()
            self.train_model(all_states, all_probs, all_results)
            print(f"Iteration {iteration + 1}/{self.num_iterations} completed.")

    def initialize_players(self):
        player1 = BetaZeroPlayer(1, self.model, verbose=False)
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
