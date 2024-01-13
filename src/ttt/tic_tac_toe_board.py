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

    # Modified to accept a tuple
    def is_valid_move(self, move):
        row, col = move
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == 0

    # Modified to accept a tuple
    def make_move(self, move, player):
        row, col = move
        if not self.is_valid_move(move):
            raise ValueError("Invalid move")

        new_board = TicTacToeBoard(self.board)
        new_board.board[row][col] = player
        return new_board

    def is_game_over(self):
        is_over, _ = self._check_game_status()
        return is_over

    def _check_game_status(self):
        for player in [1, -1]:
            if any(np.all(self.board[row, :] == player) for row in range(3)) or \
               any(np.all(self.board[:, col] == player) for col in range(3)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return True, player  # player wins
        if np.all(self.board != 0):
            return True, 0  # draw
        return False, None  # game is ongoing

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def get_game_result(self):
        _, result = self._check_game_status()
        return result

def main():
    board = TicTacToeBoard()
    # Modified to pass tuples
    board = board.make_move((0, 0), 1)
    board = board.make_move((0, 1), -1)
    board = board.make_move((0, 2), 1)
    print(board)
    print(repr(board))
    print(board.is_valid_move((0, 0)))  # Modified to pass tuple
    print("Game Result:", board.get_game_result())

if __name__ == "__main__":
    main()
