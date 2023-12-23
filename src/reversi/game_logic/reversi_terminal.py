import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import numpy as np
from reversi_board import ReversiBoard
from players.reversi_players import RandomPlayer, HumanPlayer

class ReversiTerminal:
    def __init__(self, player1, player2, size=8):
        self.board = ReversiBoard(size=size)
        self.players = {1: player1, -1: player2}
        self.current_player = 1

    def play(self):
        game_over = False

        while not game_over:
            print(self.board, "\n")
            player = self.players[self.current_player]
            moves = self.board.generate_possible_moves(self.current_player)

            if moves:
                action_matrix = player.get_move(self.board)
                try:
                    self.board = self.board.make_move(action_matrix, self.current_player)
                except ValueError as e:
                    print(e)
                    continue
            else:
                print(f"Player {'X' if self.current_player == 1 else 'O'} cannot move.")

            game_over = self.board.is_game_over()
            self.current_player *= -1

        print(self.board)
        print("Game over!")
        # Counting the disks
        count_player1 = np.count_nonzero(self.board.board == 1)
        count_player2 = np.count_nonzero(self.board.board == -1)
        if count_player1 > count_player2:
            print("Player X wins!")
        elif count_player2 > count_player1:
            print("Player O wins!")
        else:
            print("It's a tie!")



def main():
    # Example game
    player1 = RandomPlayer(1)
    player2 = RandomPlayer(-1)
    game = ReversiTerminal(player1, player2, size=3)
    game.play()

if __name__ == "__main__":
    main()