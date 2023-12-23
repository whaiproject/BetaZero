import numpy as np

class ReversiBoard:
    def __init__(self, board=None, size=8):
        if board is None:
            self.size = size
            self.board = np.zeros((size, size), dtype=int)
            # Starting position
            initial_pos = self.size // 2 - 1
            self.board[initial_pos][initial_pos], self.board[initial_pos + 1][initial_pos + 1] = 1, 1
            self.board[initial_pos][initial_pos + 1], self.board[initial_pos + 1][initial_pos] = -1, -1
        else:
            self.board = np.copy(board.board)
            self.size = int(board.size)

    def __str__(self):
        out_str = "  " + " ".join(map(str, range(self.size))) + "\n"
        for i, row in enumerate(self.board):
            out_str += str(i) + ' ' + ' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row]) + "\n"
        return out_str

    def __repr__(self):
        return f"{self.board}"

    def is_valid_move(self, row, col, player):
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row][col] != 0:
            return False

        # Check all directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid = False
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opposite = False
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == -player:
                r, c = r + dr, c + dc
                found_opposite = True
            if found_opposite and 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                valid = True
                break
        return valid

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            raise ValueError("Invalid move")

        new_board = ReversiBoard(self)
        new_board.board[row][col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            cells_to_flip = []
            while 0 <= r < self.size and 0 <= c < self.size and new_board.board[r][c] == -player:
                cells_to_flip.append((r, c))
                r, c = r + dr, c + dc
            if 0 <= r < self.size and 0 <= c < self.size and new_board.board[r][c] == player:
                for r, c in cells_to_flip:
                    new_board.board[r][c] = player
        return new_board

    def is_game_over(self):
        for player in [1, -1]:
            if any(self.is_valid_move(i, j, player) for i in range(self.size) for j in range(self.size)):
                return False
        return True

    def generate_possible_moves(self, player):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.is_valid_move(i, j, player)]



def main():
    board = ReversiBoard(size=4)
    board = board.make_move(0, 2, 1)
    board = board.make_move(0, 1, -1)
    board = board.make_move(2, 0, 1)
    print(board)
    print(repr(board))
    print(board.is_valid_move(0, 3, -1))

if __name__ == "__main__":
    main()