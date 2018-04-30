import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

input_size = 80 * 80
hidden_size = 100


class PolicyGradient(nn.Module):
    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 3)
        if torch.cuda.is_available():
            self.cuda()
        self.actions = []
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                # "Xavier" initialization
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        h = F.relu(self.hidden(x))
        logits = F.relu(self.out(h))
        probabilities = F.softmax(logits)
        return probabilities

    def reset(self):
        self.actions = []
