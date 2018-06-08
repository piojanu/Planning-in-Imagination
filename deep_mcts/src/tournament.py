from humblerl import Callback


class Tournament(Callback):
    def __init__(self):
        self.wins, self.losses, self.draws = 0, 0, 0

    def on_reset(self, train_mode):
        pass

    def on_step(self, transition, info):
        if transition.is_terminal:
            if transition.reward == 0:
                self.draws += 1
            elif transition.reward > 0:
                self.wins += 1
            else:
                self.losses += 1

    def get_results(self):
        return self.wins, self.losses, self.draws
