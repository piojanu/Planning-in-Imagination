import time

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys


def replay_buffer(buffer_dir="buffer.npz", interval=0.001):
    npzfile = np.load(buffer_dir)
    for state, action in zip(npzfile["states"], npzfile["actions"]):
        plt.imshow(state)
        plt.show(block=False)
        plt.pause(interval)


def adjust_learning_rate(optimizer, init_lr, gamma, global_step, decay_interval):
    num_decays = global_step//decay_interval
    lr = init_lr * gamma**num_decays
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def eval_in_batches(model, dataset, eval_size=1000):
    if eval_size > dataset.num_samples:
        eval_size = dataset.num_samples
    num_batches = dataset.num_samples//eval_size
    loss_accumulated = 0

    for i in range(num_batches):
        states_batch, next_states_batch, actions_batch = dataset.get_next_batch(eval_size)
        predictions = model(torch.Tensor(states_batch), torch.Tensor(actions_batch))
        loss = F.mse_loss(predictions, torch.Tensor(next_states_batch))

        loss_accumulated += loss.item()

    return loss_accumulated/num_batches


def test_model(model, dataset, interval=0.003, concat=True):
    fig = plt.figure(figsize=(1, 3))
    model = torch.load("{}.model".format(model.name))
    for _ in range(dataset.num_samples):
        states, next_states, actions = dataset.get_next_batch(1)
        prediction = model(torch.unsqueeze(torch.Tensor(states[0]), 0), torch.unsqueeze(torch.Tensor(actions[0]), 0))

        fig.add_subplot(1, 3, 1)
        # if concat:
        #     plt.imshow(state[3].reshape(80, 80).astype(np.float32))
        # else:
        #     plt.imshow(state.reshape(80, 80).astype(np.float32))
        plt.imshow(states[0].reshape(80, 80))
        fig.add_subplot(1, 3, 2)
        plt.imshow(prediction.detach().cpu().numpy().reshape(80, 80))
        fig.add_subplot(1, 3, 3)
        plt.imshow(next_states[0].reshape(80, 80).astype(np.float32))

        plt.show(block=False)

        plt.pause(interval)
