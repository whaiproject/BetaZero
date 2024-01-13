import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from tic_tac_toe import TicTacToeHeadless
from ttt.players import HumanPlayer, AIPlayer, RandomPlayer, OptimalPlayer
import numpy as np
import torch
import pandas as pd

def process_game_positions(positions):
    # Convert positions to a PyTorch tensor
    positions_tensor = torch.from_numpy(np.stack(positions))

    # Adjust positions for player turn
    for i in range(positions_tensor.shape[0]):
        positions_tensor[i] *= (-1) ** i

    # Calculate actions based on position changes
    actions = -positions_tensor[1:] - positions_tensor[:-1]

    return positions_tensor[:-1], actions

def collect_game_data(num_games, player1, player2):
    all_states = []
    all_actions = []

    for _ in range(num_games):
        game_manager = TicTacToeHeadless(player1, player2)

        raw_positions, _ = game_manager.play()
        states, actions = process_game_positions(raw_positions)

        all_states.extend(states)
        all_actions.extend(actions)

    return all_states, all_actions

def save_to_csv(states, actions, filename='tic_tac_toe_data.csv'):
    # Flatten the states and actions for CSV format
    flattened_states = [state.flatten().numpy() for state in states]
    flattened_actions = [action.flatten().numpy() for action in actions]

    print("Num. states generated: ", len(flattened_states))

    # Create a DataFrame
    df = pd.DataFrame({
        'State': [' '.join(map(str, state)) for state in flattened_states],
        'Action': [' '.join(map(str, action)) for action in flattened_actions]
    })

    # Save to CSV
    df.to_csv(filename, index=False)

# Example usage
num_games = 4
player1 = OptimalPlayer(1)
player2 = OptimalPlayer(-1)
all_states, all_actions = collect_game_data(num_games, player1, player2)

save_to_csv(all_states, all_actions)


