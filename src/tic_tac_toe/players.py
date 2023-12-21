from abc import ABC, abstractmethod
from tic_tac_toe_board import TicTacToeBoard 
import random

class Player(ABC):
    @abstractmethod
    def get_move(self, board):
        pass

class HumanPlayer(Player):
    def get_move(self, board):
        while True:
            try:
                row = int(input("Enter the row (0-2): "))
                col = int(input("Enter the column (0-2): "))
                if 0 <= row < 3 and 0 <= col < 3 and board.board[row][col] == 0:
                    return row, col
                else:
                    print("Invalid move. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 2.")

class AIPlayer(Player):
    def get_move(self, board):
        # Implement AI logic here. For now, we'll return the first available move.
        for i in range(3):
            for j in range(3):
                if board.board[i][j] == 0:
                    return i, j
                
# Useful for baseline
class RandomPlayer(Player):
    def get_move(self, board):
        return random.choice(board.generate_possible_moves())

# Optimal player
class OptimalPlayer:
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

class DummyPlayer(Player):
    def get_move(self, board):
        return random.choice(board.generate_possible_moves())