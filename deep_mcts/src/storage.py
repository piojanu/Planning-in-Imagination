from humblerl import Callback


class Storage(Callback):
    def __init__(self):
        self.small_bag = []
        self.big_bag = []

    def on_reset(self, train_mode):
        pass

    def on_step(self, transition, info):
        small_package = transition.player, transition.state, info, transition.reward
        self.small_bag.append(small_package)
        if transition.is_terminal:
            for package in self.small_bag:
                big_package = package[1], package[2], package[3] * (-1 if package[0] == 1 else 1)
                self.big_bag.append(big_package)

            self.small_bag.clear()
