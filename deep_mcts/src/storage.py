from humblerl import Callback
import numpy as np
import os
import sys
from pickle import Pickler, Unpickler
from collections import deque


class Storage(Callback):
    def __init__(self, params={}):
        """
        Storage with train examples.

        Args:
            params (JSON Dictionary):
                * 'big_bag_size' (int):      max size of big bag, when big bag is full then oldest element is removed.
                                             (Default: 1000)
                * 'store_dir' (string):      folder where to store big bag. (Default: "transitions")
                * 'store_filename' (string): filename of stored data (Default: "big_bag.examples")
                * 'load_dir' (string):       folder where to load big bag. (Default: "transitions")
                * 'load_filename' (string):  filename of loaded data (Default: "big_bag.examples")
        """
        self.params = params
        self.small_bag = deque()
        self.big_bag = deque(maxlen=params.get("big_bag_size", 1000))

    def on_reset(self, train_mode):
        self.small_bag.clear()

    def on_step(self, transition, info):
        small_package = transition.state, info, transition.reward
        self.small_bag.append(small_package)
        if len(self.small_bag) == self.small_bag.maxlen:
            self.small_bag.popleft()
        if transition.is_terminal:
            for package in self.small_bag:
                big_package = package[0], package[1], transition.reward
                if len(self.big_bag) == self.big_bag.maxlen:
                    self.big_bag.popleft()
                self.big_bag.append(big_package)

            big_package = transition.next_state, np.zeros(len(info)), transition.reward
            if len(self.big_bag) == self.big_bag.maxlen:
                self.big_bag.popleft()
            self.big_bag.append(big_package)
            self.small_bag.clear()

    def store(self):
        folder = self.params.get("store_dir", "transitions")
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = self.params.get("store_filename", "big_bag.examples")
        filename = os.path.join(folder, filename)
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        folder = self.params.get("load_dir", "transitions")
        filename = self.params.get("load_filename", "big_bag.examples")
        examples_file = os.path.join(folder, filename)
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examples_file, "rb") as f:
                self.big_bag = Unpickler(f).load()
