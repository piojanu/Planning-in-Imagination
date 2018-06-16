import torch, torch.optim as optim, torch.nn.functional as F
import argparse
import time
import numpy as np
from models import GenerativeModelMini
from utils import eval_in_batches, test_model
from dataset import get_data


def run_eval(data, model):
    restored_model = torch.load("{}.model".format(model.name))
    if torch.cuda.is_available():
        restored_model.cuda()

    test_loss = eval_in_batches(restored_model, data.test_set)
    val_loss = eval_in_batches(restored_model, data.valid_set)
    train_loss = eval_in_batches(restored_model, data.train_set)

    print("Model evaluation: test_loss: {:.5f}, val_loss: {:.5f}, train_loss: {:.5f}"
          .format(test_loss, val_loss, train_loss))

    test_model(restored_model, data.test_set)


def run_training(data, model, batch_size=100, num_epochs=1000, init_lr=0.0015, patience=30, eval_size=1000,
                 reward_loss_factor=0.001, log_step=100):
    num_steps = data.train_set.num_samples // batch_size
    intervals_without_improvement = 0

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)

    # Uncomment to calculate reference loss from scratch
    # Calculating reference loss as mean loss between state and next state
    # ref_loss_accumulated = 0
    # num_evals = data.train_set.num_samples // eval_size
    # for _ in range(num_evals):
    #     states, next_states, _ = data.train_set.get_next_batch(eval_size)
    #     loss = F.mse_loss(torch.Tensor(states), torch.Tensor(next_states))
    #     ref_loss_accumulated += loss.item()
    # reference_loss = ref_loss_accumulated / num_evals
    reference_loss = 0.0036

    # Placeholders for losses history for plots
    val_loss_hist = []
    train_loss_hist = []
    steps_hist = []

    for epoch in range(num_epochs):
        # Performing crossvaldiation
        # adjust_learning_rate(optimizer, init_lr, gamma=0.98, global_step=global_step, decay_interval=100)

        train_time = time.time()
        state_losses = []
        reward_losses = []
        losses = []
        for step in range(num_steps):
            # Training
            states_batch, next_states_batch, actions_batch, rewards_batch = data.train_set.get_next_batch(batch_size)
            states_train_tensor = torch.Tensor(states_batch)
            next_states_train_tensor = torch.Tensor(next_states_batch)
            actions_train_tensor = torch.Tensor(actions_batch)
            rewards_train_tensor = torch.Tensor(rewards_batch)

            assert states_train_tensor.shape[0] == next_states_train_tensor.shape[0], \
                "states and next states shape does not match!"

            next_states_pred, rewards_pred = model(states_train_tensor, actions_train_tensor)
            state_loss = F.mse_loss(next_states_pred, next_states_train_tensor)
            reward_loss = reward_loss_factor * F.mse_loss(rewards_pred, rewards_train_tensor)
            loss = state_loss + reward_loss

            state_losses.append(state_loss.item())
            reward_losses.append(reward_loss.item())
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_step == 0:
                print("Step {}/{}: loss: {:.5f}, state_loss: {:.5f}, reward_loss: {:.5f}"
                      .format(step+1, num_steps, np.mean(losses), np.mean(state_losses), np.mean(reward_losses)))

        eval_time = time.time()
        train_loss = np.mean(losses)
        val_loss = eval_in_batches(model, data.valid_set, reward_loss_factor)
        eval_time = time.time() - eval_time
        global_step = num_steps * (epoch + 1)
        print("Epoch: {}, step: {}, loss: [train: {:.5f}, val: {:.5f}, ref: {:.5f}], lr: {}"
              .format(epoch+1, global_step, train_loss, val_loss, reference_loss, optimizer.param_groups[0]["lr"]))
        steps_hist.append(global_step)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_loss == min(val_loss_hist):
            print("val loss improved to {:.5f}. Saving model to {}".format(val_loss, "{}.model".format(model.name)))
            torch.save(model, "{}.model".format(model.name))
            intervals_without_improvement = 0
        else:
            intervals_without_improvement += 1

        if intervals_without_improvement == patience:
            break
        scheduler.step()
    train_time = time.time() - train_time
    print("Eval time: {}. train time {}".format(eval_time, train_time))

    print("Training finished, min val loss: {}, reference loss: {}, train loss: {}".format(min(val_loss_hist),
                                                                                           reference_loss, train_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset", default="buffer100k.npz")
    parser.add_argument("--model", default="concat-input")
    args = parser.parse_args()

    # Creating model
    context = 1
    if args.model == "concat-input":
        context = 4

    data = get_data(args.dataset, context=context)
    model = GenerativeModelMini(input_channels=context, action_space=data.action_space)
    print("Action space: {}".format(data.action_space))

    print("Using {}".format(args.model))

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


