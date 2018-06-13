import torch, torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


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


def load_data(dataset_path, train_size=90000, concat=True):

    print("Loading data...")
    data = np.load(dataset_path)
    print("Loaded data.")

    states = data["states"].reshape((-1, 1, 80, 80))

    # Normalizing states
    states = (states - 127.5)/127.5

    # Transforming actions to one-hot form
    actions = data["actions"]
    actions_one_hot = np.zeros((actions.shape[0], 4))
    for action_id, action in enumerate(actions):
        actions_one_hot[action_id, action] = 1

    if concat:
        states, next_states, actions_one_hot = prepare_dataset_concat_input(states, actions_one_hot)
    else:
        states, next_states, actions_one_hot = prepare_dataset(states, actions_one_hot)

    # Dividing data into train, val sets
    data = {"train": {}, "val": {}}
    data["train"]["states"] = states[:train_size]
    data["train"]["next_states"] = next_states[:train_size]
    data["train"]["actions"] = actions_one_hot[:train_size]
    data["val"]["states"] = states[train_size:]
    data["val"]["next_states"] = next_states[train_size:]
    data["val"]["actions"] = actions_one_hot[train_size:]
    return data


def prepare_dataset(states, actions_one_hot):
    next_states = states[1:]
    states = states[:-1].astype(np.float16)
    actions_one_hot = actions_one_hot[:-1]
    assert states.shape[0] == actions_one_hot.shape[0], "Number of states and actions is inconsistent"
    return states, next_states, actions_one_hot


def prepare_dataset_concat_input(states, actions_one_hot):
    next_states = states[4:].astype(np.float16)
    states = states[:-1].astype(np.float16)
    states_concat = np.zeros((states.shape[0]-3, 4, 80, 80), dtype=np.float16)
    for i in range(states_concat.shape[0]):
        states_concat[i] = states[i:i+4].reshape((1, 4, 80, 80))
    actions_one_hot = actions_one_hot[3:-1]
    assert states_concat.shape[0] == actions_one_hot.shape[0], "Number of states and actions is inconsistent"
    return states_concat, next_states, actions_one_hot


def eval_in_batches(model, states, next_states, actions=None, eval_size=100, concat=True):
    num_batches = states.shape[0]//eval_size
    loss_accumulated = 0

    for eval_iter in range(num_batches):
        states_batch = states[eval_iter*eval_size:(eval_iter+1)*eval_size]
        next_states_batch = next_states[eval_iter * eval_size:(eval_iter + 1) * eval_size]
        if actions is not None:
            actions_batch = actions[eval_iter * eval_size:(eval_iter + 1) * eval_size]
            predictions = model(torch.Tensor(states_batch), torch.Tensor(actions_batch))
        else:
            predictions = model(torch.Tensor(states_batch))
        loss = F.mse_loss(predictions, torch.Tensor(next_states_batch))

        loss_accumulated += loss.item()

    return loss_accumulated/num_batches


def test_model(model, states, next_states, actions, interval=0.003, concat=True):
    fig = plt.figure(figsize=(1, 3))
    model = torch.load("{}.model".format(model.name))
    for state, next_state, action in zip(states, next_states, actions):
        prediction = model(torch.unsqueeze(torch.Tensor(state), 0), torch.unsqueeze(torch.Tensor(action), 0))

        fig.add_subplot(1, 3, 1)
        if concat:
            plt.imshow(state[3].reshape(80, 80).astype(np.float32))
        else:
            plt.imshow(state.reshape(80, 80).astype(np.float32))
        fig.add_subplot(1, 3, 2)
        plt.imshow(prediction.detach().cpu().numpy().reshape(80, 80))
        fig.add_subplot(1, 3, 3)
        plt.imshow(next_state.reshape(80, 80).astype(np.float32))

        plt.show(block=False)

        plt.pause(interval)
