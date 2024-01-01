from abc import ABC, abstractmethod
from ttt.tic_tac_toe_board import TicTacToeBoard 
import random
import torch
#from SL import neural_networks

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

# Useful for baseline
class RandomPlayer(Player):
    def get_move(self, board):
        return random.choice(board.get_valid_moves())

# Optimal player
class OptimalPlayer:
    def __init__(self, symbol):
        self.symbol = symbol  # 1 for X, -1 for O

    def get_move(self, board):
        if len(board.get_valid_moves()) == 9:
            return random.choice(board.get_valid_moves())
        else:
            _, best_move = self.minimax(board, True)
            return best_move

    def minimax(self, board, is_maximizing):
        game_over = board.is_game_over()
        if game_over:
            winner = board.get_game_result()
            if winner == self.symbol:
                return 1, None
            elif winner == -self.symbol:
                return -1, None
            else:
                return 0, None

        if is_maximizing:
            best_score = float("-inf")
            best_move = None
            for move in board.get_valid_moves():
                new_board = board.make_move(move, self.symbol)
                score, _ = self.minimax(new_board, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            best_score = float("inf")
            best_move = None
            for move in board.get_valid_moves():
                new_board = board.make_move(move, -self.symbol)
                score, _ = self.minimax(new_board, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

class DummyPlayer(Player):
    def get_move(self, board):
        return random.choice(board.get_valid_moves())
    
# class AIPlayer(Player):
#     def __init__(self, path_to_model, symbol):
#         super().__init__()
#         self.model = torch.load(path_to_model)
#         self.model.eval()
#         self.symbol = symbol
#         # /Users/lb1223/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Desktop/whaiproject/BetaZero/src/tic_tac_toe/SL/models/tic_tac_toe_model_2023-12-23_13-26-47.pth

#     def get_move(self, board):
#         state = self.symbol * torch.tensor(board.board).float()
#         actions = self.model(state.flatten()).reshape((3, 3))
#         rows, cols = actions.size()

#         actions = actions.flatten()

#         # Step 2: Sort the flattened tensor in descending order
#         sorted_values, sorted_indices = torch.sort(actions, descending=True)

#         # Step 3: Map the sorted indices back to the original 2D shape
#         actions = [((idx // cols).item(), (idx % cols).item()) for idx in sorted_indices]
#         valid_actions = board.get_valid_moves()

#         action = [a for a in actions if a in valid_actions][0]

#         print(action)
#         print(actions)
#         print(valid_actions)
#         #action = (self.symbol * torch.nonzero(action)).tolist()[0]
#         return action