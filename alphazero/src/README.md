# Value-function MCTS and AlphaZero

All the configuration can be found in `config.json.dist`. Copy it to `config.json` and edit. `python main.py` to run training. Look into code's docstrings for more documentation.

## In order to change between Value-function MCTS and AlphaZero you need:
1. In `main.py` change import between:
   * `from algos.value_function import build_keras_nn, Planner` and
   * `from algos.alphazero import build_keras_nn, Planner`.
2. Also in `main.py` change targets that are passed to training:
   * `current_net.train(data=np.array(boards_input), targets=np.array(target_values))`.
   * `current_net.train(data=np.array(boards_input), targets=[np.array(target_pis), np.array(target_values)])` or
3. In `.json` config change loss between:
   * `"mean_squared_error"` and
   * `["categorical_crossentropy", "mean_squared_error"]`.
