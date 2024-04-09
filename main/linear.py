import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessPredictorLinear(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=128, output_size=1, device='cpu'):
        super(SuccessPredictorLinear, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # Output layer
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear4 = nn.Linear(hidden_layer_size, output_size)

        self.activation = torch.nn.Sigmoid()

        self.device = device

    def forward(self, input_seq):
        x = self.linear1(input_seq)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        x = self.activation(x)

        return x

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
