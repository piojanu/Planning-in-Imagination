from humblerl import Callback
import numpy as np


class Storage(Callback):
    def __init__(self):
        self.small_bag = []
        self.big_bag = []

    def on_reset(self, train_mode):
        pass

    def on_step(self, transition, info):
        small_package = transition.state, info, transition.reward
        self.small_bag.append(small_package)
        if transition.is_terminal:
            for package in self.small_bag:
                big_package = package[0], package[1], transition.reward
                self.big_bag.append(big_package)

            big_package = transition.next_state, np.zeros(len(info)), transition.reward
            self.big_bag.append(big_package)
            self.small_bag.clear()
