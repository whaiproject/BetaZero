from tic_tac_toe import TicTacToeTerminal
from players import HumanPlayer, AIPlayer, RandomPlayer, OptimalPlayer

# Initialize players (modify as needed)
player1 = OptimalPlayer(1)  # or AIPlayer()
player2 = HumanPlayer()  # or AIPlayer()

# Initialize the terminal game manager
terminal_game_manager = TicTacToeTerminal(player1, player2)
terminal_game_manager.play()

# test