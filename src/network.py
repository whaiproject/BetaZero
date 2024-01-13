import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = 3 * 3  # Assuming the input is a 3x3 tensor

        # Define layers for linear_relu_stack
        self.linear1 = nn.Linear(self.input_size, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)

        # Define layers for value_network
        self.relu_value = nn.ReLU()
        self.linear_value = nn.Linear(32, 1)

        # Define layers for policy_network
        self.relu_policy = nn.ReLU()
        self.linear_policy = nn.Linear(32, 9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.linear1(x))
        logits = self.linear2(x)

        value = self.relu_value(self.linear_value(logits))
        actions = self.relu_policy(self.linear_policy(logits))
        
        value = torch.tanh(value)
        actions = F.softmax(actions, dim=1)

        return actions, value
    

class ConvolutionalTicTacToeNetWithDropout(nn.Module):
    def __init__(self):
        super(ConvolutionalTicTacToeNetWithDropout, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Flattening layer for transition from convolutional layers to linear layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output layers
        self.fc_probs = nn.Linear(128, 9)  # For 9 positions on the Tic-Tac-Toe board
        self.fc_value = nn.Linear(128, 1)  # For value estimation

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Apply dropout
        #x = self.dropout(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        #x = self.dropout(x) # rimuoverei sta qua
        x = F.relu(self.fc2(x))
        #x = self.dropout(x) # rimuoverei sta qua

        # Output layers
        probs = F.softmax(self.fc_probs(x), dim=1)
        value = torch.tanh(self.fc_value(x))

        return probs, value