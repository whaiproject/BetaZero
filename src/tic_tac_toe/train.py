import pandas as pd
import matplotlib.pyplot as plt
import random
from neural_networks import TicTacToeNet
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from datetime import datetime
import os

class TicTacToeDataset(Dataset):
    def __init__(self, csv_file, state_shape, action_shape):
        self.original_dataframe = pd.read_csv(csv_file)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dataframe = self.expand_with_transforms()

    def tensor_to_str(self, tensor):
        return ','.join(map(str, tensor.reshape(-1).tolist()))

    def expand_with_transforms(self):
        unique_data = set()
        expanded_data = []
        transforms = [
            lambda x: x,  # No transformation
            lambda x: x.flip(dims=[0]),  # Flip vertically
            lambda x: x.flip(dims=[1]),  # Flip horizontally
            lambda x: x.rot90(1, [0, 1]),  # Rotate 90 degrees
            lambda x: x.rot90(2, [0, 1]),  # Rotate 180 degrees
            lambda x: x.rot90(3, [0, 1]),  # Rotate 270 degrees
            lambda x: x.t(),  # Transpose (flip along one diagonal)
            lambda x: x.flip(dims=[0]).t(),  # Flip around the other diagonal
        ]

        for _, row in self.original_dataframe.iterrows():
            state = torch.tensor([float(x) for x in row['State'].split()], dtype=torch.float).view(self.state_shape)
            action = torch.tensor([float(x) for x in row['Action'].split()], dtype=torch.float).view(self.action_shape)
            for transform in transforms:
                transformed_state = transform(state)
                transformed_action = transform(action)

                # Check for uniqueness
                state_str = self.tensor_to_str(transformed_state)
                action_str = self.tensor_to_str(transformed_action)
                if (state_str, action_str) not in unique_data:
                    unique_data.add((state_str, action_str))
                    expanded_data.append((transformed_state, transformed_action))

        return expanded_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        state, action = self.dataframe[idx]
        return state.clone(), action.clone()


state_shape = (3, 3)  # Example for a 3x3 Tic Tac Toe board
action_shape = (3, 3) # actions are also represented in a 3x3 structure

# Parameters
validation_split = 0.2  # Percentage of data for validation
batch_size = 128
val_batch_size = 1000

# Create the full dataset
full_dataset = TicTacToeDataset('tic_tac_toe_data.csv', state_shape, action_shape)

# Splitting dataset into train and validation
total_size = len(full_dataset)
val_size = int(total_size * validation_split)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Creating data loaders for both sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)


def train_model(model, train_loader, val_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Two subplots

    epoch_pbar = tqdm(range(epochs), desc="Overall Progress", unit="epoch")
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for states, actions in train_loader:
            states = states.view(states.size(0), -1)
            actions = actions.view(actions.size(0), -1).argmax(dim=1)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += actions.size(0)
            correct_train += (predicted == actions).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.view(states.size(0), -1)
                actions = actions.view(actions.size(0), -1).argmax(dim=1)

                outputs = model(states)
                loss = criterion(outputs, actions)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += actions.size(0)
                correct_val += (predicted == actions).sum().item()

        # Calculate average losses and accuracies
        train_loss_avg = train_loss / len(train_loader)
        train_acc = correct_train / total_train
        val_loss_avg = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Update progress bar description
        epoch_pbar.set_postfix({
            'Train Acc': f'{train_acc:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Train Loss': f'{train_loss_avg:.4f}',
            'Val Loss': f'{val_loss_avg:.4f}'
        })

        # Update plots
        ax1.clear()
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        ax2.clear()
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.draw()
        plt.pause(0.001)

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Example usage
input_size = 9  # 3x3 Tic Tac Toe board flattened
hidden_size = 256 #64  # Example size of hidden layer
# with 64 neurons it wasn't able to generalize well on the validation set
# with 20 it's better
output_size = 9  # Assuming 3x3 action space

model = TicTacToeNet(input_size, hidden_size, output_size)

# Assuming you have created 'dataloader' from TicTacToeDataset
epochs = 500  # Number of epochs to train
learning_rate = 0.0001  # Learning rate

print("Length of train dataset: ", len(train_dataset))
print("Length of val dataset: ", len(val_dataset))
print("Training model...")
train_model(model, train_loader, val_loader, epochs, learning_rate)
print("Done!")


# Save the model

# Current timestamp
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Model filename with current timestamp
model_filename = f'src/tic_tac_toe/models/tic_tac_toe_model_{current_timestamp}.pth'

# Create the 'models' directory if it doesn't exist
if not os.path.exists('src/tic_tac_toe/models/'):
    os.makedirs('src/tic_tac_toe/models/')

# Assuming 'model' is your trained model instance
torch.save(model, model_filename)