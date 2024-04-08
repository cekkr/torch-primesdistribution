import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessPredictorLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=128, output_size=1, device='cpu'):
        super(SuccessPredictorLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM layer: input_size is 7 because each timestep of our sequence has 7 features
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Output layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.device = device

    def forward(self, input_seq):
        # Initializing hidden state for first input using method defined below
        hidden_cell = (torch.zeros(1, input_seq.shape[0], self.hidden_layer_size).to(device=self.device),
                       torch.zeros(1, input_seq.shape[0], self.hidden_layer_size).to(device=self.device))

        # Forward pass through LSTM layer
        # lstm_out shape: (batch_size, seq_length, hidden_layer_size)
        lstm_out, hidden_cell = self.lstm(input_seq, hidden_cell)

        # Only take the output from the final timestep
        # You can modify this part to use outputs from different timesteps
        lstm_out = lstm_out[:, -1, :]

        # Pass through the output layer and apply sigmoid activation to get the probability
        predictions = torch.sigmoid(self.linear(lstm_out))

        return predictions

    def save(self, where):
        torch.save(self.state_dict(), '%s/model.pth' % (where))

# Assume `input_array` is your input numpy array of shape (batch_size, 200, 7)
# Convert the numpy array to a PyTorch tensor
# input_tensor = torch.from_numpy(input_array).float()

# Instantiate the model
# model = SuccessPredictorLSTM()

# Forward pass to get the probability
# Note: No need to unsqueeze if you already have a batch dimension
# probability = model(input_tensor)
