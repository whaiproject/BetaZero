import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import numpy as np
from reversi_board import ReversiBoard
from players.reversi_players import RandomPlayer, HumanPlayer
import tkinter as tk
from tkinter import messagebox
import numpy as np

# Import your ReversiBoard and player classes here
# from your_reversi_modules import ReversiBoard, HumanPlayer, RandomPlayer

class ReversiGUI:
    def __init__(self, player1, player2, size=8):
        self.board = ReversiBoard(size=size)
        self.players = {1: player1, -1: player2}
        self.current_player = 1

        self.root = tk.Tk()
        self.root.title("Reversi")
        self.buttons = [[None for _ in range(size)] for _ in range(size)]
        self.initialize_board()

    def initialize_board(self):
        for i in range(self.board.size):
            for j in range(self.board.size):
                self.buttons[i][j] = tk.Button(self.root, height=2, width=4, 
                                               command=lambda row=i, col=j: self.on_button_click(row, col))
                self.update_button_text(i, j)
                self.buttons[i][j].grid(row=i, column=j)

    def on_button_click(self, row, col):
        try:
            moves = self.board.generate_possible_moves(self.current_player)
            if (row, col) in moves:
                self.board = self.board.make_move(row, col, self.current_player)
                self.update_board()
                
                game_over = self.board.is_game_over()
                if game_over:
                    self.end_game()
                else:
                    self.current_player *= -1
                    self.handle_next_player()
            else:
                messagebox.showerror("Invalid Move", "This move is not allowed.")

        except ValueError as e:
            messagebox.showerror("Invalid Move", str(e))

    def handle_next_player(self):
        moves = self.board.generate_possible_moves(self.current_player)
        if not moves:  # No valid moves for the current player
            if self.board.is_game_over():  # Check if the game is over
                self.end_game()
            else:
                self.current_player *= -1  # Skip turn
                self.handle_next_player()  # Recursively handle the next player
        elif not isinstance(self.players[self.current_player], HumanPlayer):
            self.ai_move()

    def ai_move(self):
        self.root.after(600, self._execute_ai_move)

    def _execute_ai_move(self):
        moves = self.board.generate_possible_moves(self.current_player)
        if moves:
            row, col = self.players[self.current_player].get_move(self.board)
            self.on_button_click(row, col)

    def end_game(self):
        count_player1 = np.count_nonzero(self.board.board == 1)
        count_player2 = np.count_nonzero(self.board.board == -1)
        if count_player1 > count_player2:
            winning_msg = "Player X wins!"
        elif count_player2 > count_player1:
            winning_msg = "Player O wins!"
        else:
            winning_msg = "It's a tie!"

        messagebox.showinfo("Game Over", winning_msg)
        self.root.destroy()

    def update_button_text(self, row, col):
        value = self.board.board[row][col]
        text = 'X' if value == 1 else 'O' if value == -1 else ''
        self.buttons[row][col]['text'] = text

    def update_board(self):
        for i in range(self.board.size):
            for j in range(self.board.size):
                self.update_button_text(i, j)

    def play(self):
        # Check if the first player is not a HumanPlayer
        if not isinstance(self.players[self.current_player], HumanPlayer):
            self.ai_move()
        self.root.mainloop()

def main():
    # Example game
    player1 = RandomPlayer(1)
    player2 = RandomPlayer(-1)
    game = ReversiGUI(player1, player2, size=4)
    game.play()

if __name__ == "__main__":
    main()
