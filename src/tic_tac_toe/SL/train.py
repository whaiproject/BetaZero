import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TicTacToeDataset(Dataset):
    def __init__(self, csv_file, state_shape, action_shape):
        self.dataframe = pd.read_csv(csv_file)
        self.state_shape = state_shape
        self.action_shape = action_shape
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        state = torch.tensor([float(x) for x in row['State'].split()], dtype=torch.float).view(self.state_shape)
        action = torch.tensor([float(x) for x in row['Action'].split()], dtype=torch.float).view(self.action_shape)
        return state, action

# Example usage
state_shape = (3, 3)  # Example for a 3x3 Tic Tac Toe board
action_shape = (3, 3) # actions are also represented in a 3x3 structure
dataset = TicTacToeDataset('tic_tac_toe_data.csv', state_shape, action_shape)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for states, actions in dataloader:
    print(states)
    print(actions)
    break