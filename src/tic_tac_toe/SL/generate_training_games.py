import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from tic_tac_toe import TicTacToeHeadless
from players import HumanPlayer, AIPlayer, RandomPlayer, OptimalPlayer
import torch

# Initialize players (modify as needed)
player1 = RandomPlayer()  # or AIPlayer()
player2 = RandomPlayer()  # or AIPlayer()

# Initialize the terminal game manager
#terminal_game_manager = TicTacToeTerminal(player1, player2)
#terminal_game_manager.play()

print("Generating games...")
game_manager = TicTacToeHeadless(player1, player2)
positions, z = game_manager.play()

print("Converting to tensors...")
positions = torch.tensor(positions)

print(z)
print(positions)