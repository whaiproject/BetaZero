import numpy as np

class TicTacToeBoard:
    def __init__(self, board=None):
        self.board = np.zeros((3, 3), dtype=int) if board is None else np.copy(board)

    def __str__(self):
        out_str = ""
        for i, row in enumerate(self.board):
            if i != 0:
                out_str += '\n'
            out_str += ' ' + ' | '.join(['X' if cell == 1 else 'O' if cell == -1 else ' ' for cell in row]) + ' '
            if i != 2:
                out_str += '\n---+---+---'
        return out_str
    
    def __repr__(self):
        return f"{self.board}"

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == 0

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col):
            raise ValueError("Invalid move")

        new_board = TicTacToeBoard(self.board)
        new_board.board[row][col] = player
        return new_board

    def is_game_over(self):
        for player in [1, -1]:
            if any(np.all(self.board[row, :] == player) for row in range(3)) or \
               any(np.all(self.board[:, col] == player) for col in range(3)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return True, player
        if np.all(self.board != 0):
            return True, 0
        return False, None

    def generate_possible_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

def main():
    board = TicTacToeBoard()
    board = board.make_move(0, 0, 1)
    board = board.make_move(0, 1, -1)
    board = board.make_move(0, 2, 1)
    print(board)
    print(repr(board))
    print(board.is_valid_move(0, 0))

if __name__ == "__main__":
    main()