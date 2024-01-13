import tkinter as tk
from tkinter import messagebox
from ttt.tic_tac_toe_board import TicTacToeBoard

class TicTacToeHeadless:
    def __init__(self, player1, player2):
        self.board = TicTacToeBoard()
        self.players = {1: player1, -1: player2}
        self.current_player = 1

    def play(self):
        game_over, winner = False, None
        game_states = []  # Store board positions after each move
        game_moves = []   # Store moves made by players

        while not game_over:
            player = self.players[self.current_player]

            # The current board state
            current_board_state = self.board.board

            # Use the player's get_move method (which uses MCTS)
            move = player.get_move(self.board)

            try:
                self.board = self.board.make_move(move, self.current_player)
                game_states.append(current_board_state)  # Store the state
                game_moves.append(move)             # Store the move
            except ValueError as e:
                raise ValueError(f"Invalid move: {e}")

            game_over = self.board.is_game_over()
            self.current_player *= -1
        
        winner = self.board.get_game_result()

        return game_states, game_moves, winner



class TicTacToeTerminal:
    def __init__(self, player1, player2, board=None):
        if board is None:
            self.board = TicTacToeBoard()
        else:
            self.board = board
        self.players = {1: player1, -1: player2}
        self.current_player = 1

    def play(self):
        game_over, winner = False, None

        while not game_over:
            print(self.board, "\n")
            player = self.players[self.current_player]
            move = player.get_move(self.board)
            
            try:
                self.board = self.board.make_move(move, self.current_player)
            except ValueError as e:
                print(e)
                continue

            game_over = self.board.is_game_over()
            self.current_player *= -1

            if game_over:
                winner = self.board.get_game_result()
                print(self.board)
                if winner == 0:
                    print("It's a tie!")
                else:
                    print(f"Player {'X' if winner == 1 else 'O'} wins!")

        print("Game over!")

class TicTacToeGUI:
    def __init__(self, player1, player2):
        self.board = TicTacToeBoard()
        self.players = {1: player1, -1: player2}
        self.current_player = 1
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.initialize_board()

    def initialize_board(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.root, height=3, width=6, 
                                               command=lambda row=i, col=j: self.on_button_click(row, col))
                self.buttons[i][j].grid(row=i, column=j)

    def on_button_click(self, row, col):
        try:
            self.board = self.board.make_move(row, col, self.current_player)
            self.buttons[row][col]['text'] = 'X' if self.current_player == 1 else 'O'
            self.buttons[row][col]['state'] = 'disabled'
            
            game_over, winner = self.board.is_game_over()
            if game_over:
                self.end_game(winner)
            else:
                self.current_player *= -1
                # Check if the next player is not a HumanPlayer
                if not isinstance(self.players[self.current_player], HumanPlayer):
                    self.ai_move()

        except ValueError as e:
            messagebox.showerror("Invalid Move", str(e))

    def ai_move(self):
        self.root.after(600, self._execute_ai_move)

    def _execute_ai_move(self):
        row, col = self.players[self.current_player].get_move(self.board)
        self.on_button_click(row, col)

    def end_game(self, winner):
        winning_msg = f"Player {'X' if winner == 1 else 'O'} wins!" if winner != 0 else "It's a tie!"
        messagebox.showinfo("Game Over", winning_msg)
        self.root.destroy()

    def play(self):
        # Check if the current player is not a HumanPlayer
        if not isinstance(self.players[self.current_player], HumanPlayer):
            self.ai_move()
        self.root.mainloop()


def main():
    # Prompt for the interface choice
    interface = input("Choose interface gui (g) or terminal(t): ").strip().lower()

    # Initialize players (modify as needed)
    player1 = AIPlayer()  # or AIPlayer()
    player2 = HumanPlayer()  # or AIPlayer()

    if interface == 'g':
        # Initialize the GUI game manager
        gui_game_manager = TicTacToeGUI(player1, player2)
        gui_game_manager.play()
    elif interface == 't':
        # Initialize the terminal game manager
        terminal_game_manager = TicTacToeTerminal(player1, player2)
        terminal_game_manager.play()
    else:
        print("Invalid interface choice.")

if __name__ == "__main__":
    main()