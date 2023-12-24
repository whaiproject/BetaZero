from tic_tac_toe import TicTacToeTerminal, TicTacToeGUI
from players import HumanPlayer, AIPlayer, RandomPlayer, OptimalPlayer

# Initialize players (modify as needed)
player2 = AIPlayer(
    path_to_model="src/tic_tac_toe/models/tic_tac_toe_model_2023-12-24_19-04-38.pth",  #"src/tic_tac_toe/SL/models/tic_tac_toe_model_2023-12-23_13-26-47.pth",
    symbol=-1
    )

player1 = OptimalPlayer(1)

# Initialize the terminal game manager
terminal_game_manager = TicTacToeTerminal(player1, player2)
terminal_game_manager.play()

# test