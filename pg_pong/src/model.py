import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils


class PolicyGradientModel(nn.Module):
    """Neural network policy gradient model.
    Contains 1 fully-connected hidden layer.
    """
    def __init__(self, input_size=6400, hidden_size=200, output_size=3,
                 init_method='xavier'):
        super(PolicyGradientModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.initialize_weights(init_method)
        self.rewards = []
        self.log_probs = []

    def initialize_weights(self, init_method):
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                if init_method == 'xavier':
                    nn.init.xavier_normal(layer.weight)
                else:
                    nn.init.normal(layer.weight)

    def forward(self, x):
        return self.model(x)

    def reset(self):
        del self.rewards[:]
        del self.log_probs[:]

    def choose_action(self, state):
        """Given current state, sample an action from model's output.

        Args:
            state (np.array): Current pong state, shape (6400,).

        Returns:
            tuple: (int) Chosen action: 1, 2 or 3,
                   (Variable) log probability of action.
        """
        state_var = Variable(torch.from_numpy(state)).type(torch.FloatTensor).unsqueeze(0)  # pylint: disable=E1101
        probs = self.forward(state_var)
        prob_dist = torch.distributions.Categorical(probs)
        action = prob_dist.sample()
        self.log_probs.append(prob_dist.log_prob(action))
        return action.data[0] + 1

    def backward(self, discount_factor):
        rws = utils.discount_rewards(self.rewards, discount_factor)
        rws = torch.FloatTensor(rws)
        rws = (rws - rws.mean()) / (rws.std() + np.finfo(np.float32).eps)  # pylint: disable=E1101
        loss = []
        for log_prob, reward in zip(self.log_probs, rws):
            loss.append(-log_prob * reward)
        loss = torch.cat(loss).sum()  # pylint: disable=E1101
        loss.backward()
        self.reset()

    def show_grads(self):
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                print(layer, layer.weight.grad)

    def show_weights(self):
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                print(layer, layer.weight)
