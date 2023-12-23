from abc import ABC, abstractmethod
import random
import time

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
                    return row, col
                else:
                    print("Invalid move. Please enter a valid row and column within the board boundaries.")
            except ValueError:
                print(f"Invalid input. Please enter a number between 0 and {board.size-1}.")

class RandomPlayer(ReversiPlayer):
    def __init__(self, symbol):
        self.symbol = symbol  # 1 for X, -1 for O

    def get_move(self, board):
        moves = board.generate_possible_moves(self.symbol)
        return random.choice(moves) if moves else (None, None)

# Optimal player
import random

class OptimalPlayer:
    def __init__(self, symbol, max_depth=4):
        self.symbol = symbol  # 1 for X, -1 for O
        self.max_depth = max_depth

    def get_move(self, board):
        _, best_move = self.minimax(board, True, 0)
        if best_move is None:
            return random.choice(board.generate_possible_moves(self.symbol))
        return best_move

    def minimax(self, board, is_maximizing, depth):
        if depth >= self.max_depth or board.is_game_over():
            return self.evaluate_board(board), None

        if is_maximizing:
            best_score = float("-inf")
            best_move = None
            for move in board.generate_possible_moves(self.symbol):
                new_board = board.make_move(*move, self.symbol)
                score, _ = self.minimax(new_board, False, depth + 1)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            best_score = float("inf")
            best_move = None
            for move in board.generate_possible_moves(-self.symbol):
                new_board = board.make_move(*move, -self.symbol)
                score, _ = self.minimax(new_board, True, depth + 1)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

    def evaluate_board(self, board):
        # Simple evaluation function that counts the difference in pieces
        winner, (count_player1, count_player2) = board.get_score()
        if self.symbol == 1:
            return count_player1 - count_player2 
        else:
            return count_player2 - count_player1
