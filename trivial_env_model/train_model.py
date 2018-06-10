import torch, torch.optim as optim, torch.nn.functional as F
from matplotlib import pyplot as plt
import argparse
from models import autoencoder, GenerativeModelMini
from utils import eval_in_batches, test_model
from dataset import get_data


def run_eval(data, model):
    restored_model = torch.load("{}.model".format(model.name))

    test_loss = eval_in_batches(restored_model, data.valid_set)
    val_loss = eval_in_batches(restored_model, data.valid_set)
    train_loss = eval_in_batches(restored_model, data.train_set)

    print("Model evaluation: test_loss: {}, val_loss: {}, train_loss: {}".format(test_loss, val_loss, train_loss))

    test_model(restored_model, data.test_set)


def run_training(data, model, batch_size=100, num_epochs=500, init_lr=0.0001, patience=10, eval_size=1000):
    num_steps = data.train_set.num_samples // batch_size
    intervals_without_improvement = 0

    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    # Calculating reference loss as mean loss between state and next state
    ref_loss_accumulated = 0
    num_evals = data.train_set.num_samples // eval_size
    for _ in range(num_evals):
        states, next_states, _ = data.train_set.get_next_batch(eval_size)
        loss = F.mse_loss(torch.Tensor(states), torch.Tensor(next_states))
        ref_loss_accumulated += loss.item()
    reference_loss = ref_loss_accumulated / num_evals

    # Placeholders for losses history for plots
    val_loss_hist = []
    train_loss_hist = []
    steps_hist = []

    for epoch in range(num_epochs):
        # Performing crossvaldiation
        global_step = num_steps * epoch
        # adjust_learning_rate(optimizer, init_lr, gamma=0.98, global_step=global_step, decay_interval=100)

        val_loss = eval_in_batches(model, data.valid_set)
        train_loss = eval_in_batches(model, data.train_set)

        print("Epoch: {}, step: {}, loss: [val: {}, train: {}, ref: {}], lr: {}".format(epoch, global_step,
                                                                                        val_loss, train_loss,
                                                                                        reference_loss,
                                                                                        optimizer.param_groups[0][
                                                                                            "lr"]))
        steps_hist.append(global_step)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_loss == min(val_loss_hist):
            print("val loss improved to {}. Saving model to {}".format(val_loss, "{}.model".format(model.name)))
            torch.save(model, "{}.model".format(model.name))
            intervals_without_improvement = 0
        else:
            intervals_without_improvement += 1

        if intervals_without_improvement == patience:
            break

        for step in range(num_steps):
            # Training
            states_batch, next_states_batch, actions_batch = data.train_set.get_next_batch(batch_size)
            states_train_tensor = torch.Tensor(states_batch)
            next_states_train_tensor = torch.Tensor(next_states_batch)
            actions_train_tensor = torch.Tensor(actions_batch)

            if states_train_tensor.shape != next_states_train_tensor.shape:
                raise Exception("states and next states shape does not match!")

            # predictions_train = model(states_train_tensor, actions_train_tensor)
            predictions_train = model(states_train_tensor, actions_train_tensor)
            loss = F.mse_loss(predictions_train, next_states_train_tensor)

            model.zero_grad()

            loss.backward()

            optimizer.step()

    print("Training finished, min val loss: {}, reference loss: {}, train loss: {}".format(min(val_loss_hist),
                                                                                           reference_loss, train_loss))

    plt.plot(steps_hist, val_loss_hist)
    plt.plot(steps_hist, train_loss_hist)
    plt.ylabel('MSE', fontsize=18)
    plt.xlabel('global step', fontsize=18)
    plt.legend(['validation error', 'train error'], fontsize=14)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset", default="buffer100k.npz")
    parser.add_argument("--model", default="generative")
    args = parser.parse_args()

    data = get_data(args.dataset)

    # Creating model
    if args.model == "generative":
        model = GenerativeModelMini()
    elif args.model == "autoencoder":
        model = autoencoder()
    else:
        raise Exception("Model has to be either generative or autoencoder")

    # Choosing HW
    if torch.cuda.is_available():
        print("Running on GPU!")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model.cuda()
    else:
        print("Running on CPU!")
        torch.set_default_tensor_type('torch.FloatTensor')

    # Running training or evaluation
    if args.train:
        run_training(data, model)
    elif args.eval:
        run_eval(data, model)
    else:
        raise Exception("Please specify either train or eval param")
    data.close()


