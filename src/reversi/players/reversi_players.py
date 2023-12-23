import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import numpy as np
import random
from abc import ABC, abstractmethod
from game_logic.reversi_board import ReversiBoard

class ReversiPlayer(ABC):
    @abstractmethod
    def get_move(self, board):
        pass

class HumanPlayer(ReversiPlayer):
    def __init__(self, symbol):
        self.symbol = symbol  # 1 for X, -1 for O

    def get_move(self, board):
        while True:
            try:
                row = int(input(f"Enter the row (0-{board.size-1}): "))
                col = int(input(f"Enter the column (0-{board.size-1}): "))
                if 0 <= row < board.size and 0 <= col < board.size and board.is_valid_move(row, col, self.symbol):
                    action_matrix = np.zeros((board.size, board.size), dtype=int)
                    action_matrix[row, col] = 1
                    return action_matrix
                else:
                    print("Invalid move. Please enter a valid row and column within the board boundaries.")
            except ValueError:
                print(f"Invalid input. Please enter a number between 0 and {board.size-1}.")

class RandomPlayer(ReversiPlayer):
    def __init__(self, symbol):
        self.symbol = symbol  # 1 for X, -1 for O

    def get_move(self, board):
        moves = board.generate_possible_moves(self.symbol)
        if moves:
            row, col = random.choice(moves)
            action_matrix = np.zeros((board.size, board.size), dtype=int)
            action_matrix[row, col] = 1
            return action_matrix
        else:
            return np.zeros((board.size, board.size), dtype=int)


# Optimal player
class OptimalPlayer(ReversiPlayer):
    def __init__(self, symbol):
        self.symbol = symbol  # 1 for X, -1 for O

    def get_move(self, board):
        if len(board.generate_possible_moves()) == 9:
            return random.choice(board.generate_possible_moves())
        else:
            _, best_move = self.minimax(board, True)
            return best_move

    def minimax(self, board, is_maximizing):
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == self.symbol:
                return 1, None
            elif winner == -self.symbol:
                return -1, None
            else:
                return 0, None

        if is_maximizing:
            best_score = float("-inf")
            best_move = None
            for move in board.generate_possible_moves():
                new_board = board.make_move(*move, self.symbol)
                score, _ = self.minimax(new_board, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            best_score = float("inf")
            best_move = None
            for move in board.generate_possible_moves():
                new_board = board.make_move(*move, -self.symbol)
                score, _ = self.minimax(new_board, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move


def main():
    # Example game
    board = ReversiBoard(size=4)
    print(board)
    player = HumanPlayer(1)
    action = player.get_move(board)
    print(action)

if __name__ == "__main__":
    main()