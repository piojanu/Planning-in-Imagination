import time
import threading

import numpy as np
import h5py
import cv2


def get_data(data_path, train_fraction=0.9, valid_fraction=0.05, context=1):
    if data_path.endswith('.hdf5'):
        data = Hdf5Data(data_path, train_fraction, valid_fraction, context)
    else:
        data = NpzData(data_path, train_fraction, valid_fraction, context)
    return data


class Data(object):
    """Generate train/valid/test sets from given data file."""

    def __init__(self, data_path, train_fraction=0.9, valid_fraction=0.05):
        self.data_path = data_path
        self.train_fraction = train_fraction
        self.valid_fraction = valid_fraction
        self.test_fraction = 1.0 - train_fraction - valid_fraction
        assert self.test_fraction > 0, "Not enough data for test set"
        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def close(self):
        pass


class NpzData(Data):
    """Generate train/valid/test sets from given .npz file.

    All data is loaded and preprocessed at once (works with small datasets).
    """

    def __init__(self, data_path, train_fraction=0.9, valid_fraction=0.05, context=1):
        super(NpzData, self).__init__(data_path, train_fraction, valid_fraction)
        print("Loading data...")
        data = np.load(data_path)
        print("Loaded data.")

        states = data["states"].reshape((-1, 1, 80, 80)).astype(np.float16)
        states = (states - 127.5) / 127.5

        states_concat = np.zeros((states.shape[0], context, 80, 80), dtype=np.float32)
        states = np.vstack([np.broadcast_to(states[0], (context-1, 1, 80, 80)), states])
        for i in range(states_concat.shape[0]):
            states_concat[i] = states[i:i + context].reshape((1, context, 80, 80))

        next_states = np.vstack([states[context:], states[-1:]])
        states = states_concat

        # Transforming actions to one-hot form
        actions = data["actions"]
        actions_one_hot = np.zeros((actions.shape[0], 4))
        for action_id, action in enumerate(actions):
            actions_one_hot[action_id, action] = 1

        data_tuple = (states, next_states, actions_one_hot)

        assert states.shape[0] == actions_one_hot.shape[0], "Number of states and actions is inconsistent"

        train_idx = int(train_fraction * states.shape[0])
        valid_idx = train_idx + int(valid_fraction * states.shape[0])
        test_idx = states.shape[0]
        self.train_set = BasicDataset(data=data_tuple, beg_idx=0, end_idx=train_idx)
        self.valid_set = BasicDataset(data=data_tuple, beg_idx=train_idx, end_idx=valid_idx)
        self.test_set = BasicDataset(data=data_tuple, beg_idx=valid_idx, end_idx=test_idx)


class Hdf5Data(Data):
    """Generate train/valid/test sets from given .hdf5 file.

    Since HDF5 files are huge and compressed, data is loaded and preprocessed dynamically.
    """

    def __init__(self, data_path, train_fraction=0.9, valid_fraction=0.05, context=1, num_traj=20):
        super(Hdf5Data, self).__init__(data_path, train_fraction, valid_fraction)
        self.data = h5py.File(data_path, "r")
        traj_len = self.data.attrs["TRAJECTORY_LEN"]
        if num_traj is None:
            num_traj = self.data.attrs["NUM_EPISODES"]
        data_size = num_traj * traj_len

        train_size = int(train_fraction * data_size)
        train_window_size = 4*traj_len if train_size > 4*traj_len else train_size
        valid_size = int(valid_fraction * data_size)
        valid_window_size = 2*traj_len if valid_size > 2*traj_len else valid_size
        test_size = data_size - train_size - valid_size
        test_window_size = 2*traj_len if test_size > 2*traj_len else test_size

        fn = Hdf5Data.get_fetch_and_preprocess_fn(context)

        print("Initializing training set...")
        self.train_set = PrefetchDataset(data=self.data, beg_idx=0, end_idx=train_size,
                                         window_size=train_window_size,
                                         fetch_and_preprocess_fn=fn, n_threads=4)
        print("Initializing validation set...")
        self.valid_set = PrefetchDataset(data=self.data, beg_idx=train_size, end_idx=train_size + valid_size,
                                         window_size=valid_window_size,
                                         fetch_and_preprocess_fn=fn, n_threads=2)
        print("Initializing test set...")
        self.test_set = PrefetchDataset(data=self.data, beg_idx=train_size + valid_size, end_idx=data_size,
                                        window_size=test_window_size,
                                        fetch_and_preprocess_fn=fn, n_threads=2)

    def close(self):
        self.data.close()

    @staticmethod
    def get_fetch_and_preprocess_fn(context):
        def fetch_and_preprocess_hdf5(data, beg, end):
            """Load and preprocess chunk of data from HDF5 file.

            Preprocessing used:
              - states:
                - resize to 80x80
                - change to gray scale
                - normalize
                - concat `context` nubmer of frames as input
              - actions:
                - change to one-hot

            Args:
                data (hdf5 file): Open HDF5 file with trajectories.
                beg (int): Beginning index of chunk to load.
                end (int): Ending index of chunk to load.

            Returns:
                tuple: States, next states, actions (one-hot).
            """
            buf_size = end - beg
            transitions = data["transition"][beg:end]
            states = np.zeros((buf_size, 1, 80, 80), dtype=np.float32)
            for i, s in enumerate(data["state"][beg:end].astype(np.float32)):
                s = cv2.resize(s, (80, 80))
                s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
                s = (s - 127.5) / 127.5
                states[i][0] = s

            states_concat = np.zeros((states.shape[0], context, 80, 80), dtype=np.float32)
            states = np.vstack([np.broadcast_to(states[0], (context-1, 1, 80, 80)), states])
            for i in range(states_concat.shape[0]):
                states_concat[i] = states[i:i + context].reshape((1, context, 80, 80))

            next_states = np.vstack([states[context:], states[-1:]])
            states = states_concat

            actions = transitions[:, 1]
            actions_one_hot = np.zeros((actions.shape[0], 4))
            for action_id, action in enumerate(actions):
                actions_one_hot[action_id, int(action)] = 1

            return states, next_states, actions_one_hot

        return fetch_and_preprocess_hdf5


class Dataset(object):
    """Holds data for given dataset. Used to generate batches."""

    def __init__(self, data, beg_idx, end_idx):
        """Initialize dataset.

        Args:
            data: Holds all data (not only for this dataset).
            beg_idx (int): Starting index of dataset in `data`.
            end_idx (int): Ending index of dataset in `data`.
        """
        self.data = data
        self.beg_idx = beg_idx
        self.end_idx = end_idx
        self.current_idx = beg_idx
        self.num_samples = self.end_idx - self.beg_idx

    def get_next_batch(self, batch_size):
        """Generate next batch of data.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            tuple: Batch of data.
        """
        raise NotImplementedError


class BasicDataset(Dataset):
    """Basic dataset object, which iterates over the already loaded data."""

    def __init__(self, data, beg_idx, end_idx):
        super(BasicDataset, self).__init__(data, beg_idx, end_idx)

    def get_next_batch(self, batch_size):
        beg = self.current_idx
        end = beg + batch_size
        data_batch = ()
        for d in self.data:
            data_batch += (d[beg:end],)
        self.current_idx = end
        if self.current_idx + batch_size > self.end_idx:
            self.current_idx = self.beg_idx
        return data_batch


class PrefetchDataset(Dataset):
    """Dataset for HDF5 file, which fetches and preprocesses data dynamically, as needed."""
    def __init__(self, data, beg_idx, end_idx, fetch_and_preprocess_fn=None, window_size=10000, n_threads=4):
        """Initialize dataset.

        Args:
            data: Holds all data (not only for this dataset).
            beg_idx (int): Starting index of dataset in `data`.
            end_idx (int): Ending index of dataset in `data`.
            fetch_and_preprocess_fn (function): Functions that loads and preprocesses a chunk of data.
            window_size (int): Window size - determines how much samples are stored in memory.
            n_threads (int): Number of threads used to prefetch and preprocess data.
                             Each thread is responsible for one particular part of data window.
        """
        super(PrefetchDataset, self).__init__(data, beg_idx, end_idx)
        self.fetch_and_preprocess_fn = fetch_and_preprocess_fn
        self.window_size = window_size
        self.window_cur_idx = 0
        self.window_data = []
        self.n_threads = n_threads
        self.prefetch_size = int(self.window_size/n_threads)
        self.prefetch_threads = []
        for i in range(n_threads):
            self._start_prefetching(i)
        self._finish_prefetching(0)

    class PrefetchThread(threading.Thread):
        def __init__(self, fn, data, beg=None, end=None):
            super(PrefetchDataset.PrefetchThread, self).__init__()
            self.fn = fn
            self.data = data
            self.beg = beg
            self.end = end
            self.new_data = None

        def run(self):
            self.new_data = self.fn(self.data, self.beg, self.end)

    def _start_prefetching(self, thread_id):
        """Prefetch chunk of data corresponding to thread of given id."""
        t = PrefetchDataset.PrefetchThread(
            self.fetch_and_preprocess_fn, self.data, beg=self.current_idx, end=self.current_idx+self.prefetch_size)
        self.current_idx += self.prefetch_size
        if self.current_idx >= self.end_idx:
            # Dataset over, start again.
            self.current_idx = self.beg_idx
        t.start()
        if len(self.prefetch_threads) == thread_id:
            self.prefetch_threads.append(t)
        else:
            self.prefetch_threads[thread_id] = t

    def _finish_prefetching(self, thread_id):
        """Receive and store preprocessed chunk of data from thread of given id."""
        t = self.prefetch_threads[thread_id]
        t.join()
        if not self.window_data:
            for i, nd in enumerate(t.new_data):
                self.window_data.append(np.vstack([nd for _ in range(self.n_threads)]))
        else:
            beg = thread_id * self.prefetch_size
            for i, wd in enumerate(self.window_data):
                wd[beg:beg+self.prefetch_size] = t.new_data[i]

    def get_next_batch(self, batch_size):
        beg = self.window_cur_idx
        end = beg + batch_size
        thread_id = beg // self.prefetch_size
        threshold = (thread_id + 1) * self.prefetch_size
        prefetched = False
        if end > threshold:
            self._finish_prefetching((thread_id + 1) % self.n_threads)
            prefetched = True

        data_batch = ()
        for d in self.window_data:
            data_batch += (d[beg:end],)
        self.window_cur_idx = end

        if end >= threshold or self.window_cur_idx + batch_size > self.window_size:
            self._start_prefetching(thread_id)
            if not prefetched:
                self._finish_prefetching((thread_id + 1) % self.n_threads)
            if self.window_cur_idx + batch_size > self.window_size:
                self.window_cur_idx = 0
        return data_batch
